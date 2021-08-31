import argparse
import math
import pprint
import sys

# This needs to happen first to avoid pyarrow serialization errors.
from pyspark.sql import SparkSession

# Make sure pyarrow is referenced before anything else to avoid segfault due to conflict
# with TensorFlow libraries.  Use `pa` package reference to ensure it's loaded before
# functions like `deserialize_model` which are implemented at the top level.
# See https://jira.apache.org/jira/browse/ARROW-3346
import pyarrow as pa

import horovod
import horovod.tensorflow.keras as hvd
import tensorflow as tf
from horovod.spark.common.backend import SparkBackend
from tensorflow.keras.layers import BatchNormalization, Input, Embedding, Concatenate, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding

PETASTORM_DATALOADER = 'petastorm'
NVTABULAR_DATALOADER = 'nvtabular'

CONTINUOUS_COLUMNS = [f'i{i}' for i in range(13)]
CATEGORICAL_COLUMNS = [f'c{c}' for c in range(26)]
ALL_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
LABEL_COLUMNS = ['clicked']


def get_category_dimensions(spark, data_dir):
    df = spark.read.csv(f'{data_dir}/dimensions/*.csv', header=True).toPandas()
    dimensions = df.to_dict('records')[0]
    pprint.pprint(dimensions)
    return dimensions


def build_model(dimensions, args):
    inputs = {
        **{i: Input(shape=(1,), name=i, dtype=tf.float32) for i in CONTINUOUS_COLUMNS},
        **{c: Input(shape=(1,), name=c, dtype=tf.int32) for c in CATEGORICAL_COLUMNS}
    }

    one_hots = []
    embeddings = []
    for c in CATEGORICAL_COLUMNS:
        dimension = int(dimensions[c]) + 1
        if dimension <= 128:
            one_hots.append(CategoryEncoding(num_tokens=dimension, name=f'one_hot_{c}')(inputs[c]))
        else:
            embedding_size = int(math.floor(0.6 * dimension ** 0.25))
            embeddings.append(Embedding(input_dim=dimension,
                                        output_dim=embedding_size,
                                        input_length=1,
                                        name=f'embedding_{c}')(inputs[c]))

    x = Concatenate(name='embeddings_concat')(embeddings)
    x = Flatten(name='embeddings_flatten')(x)
    x = Concatenate(name='inputs_concat')([x] + one_hots + [inputs[i] for i in CONTINUOUS_COLUMNS])
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)
    model = tf.keras.Model(inputs=[inputs[c] for c in ALL_COLUMNS], outputs=output)
    if hvd.rank() == 0:
        model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    opt = hvd.DistributedOptimizer(opt)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

    return model


def train_fn(dimensions, train_rows, val_rows, args):
    # Make sure pyarrow is referenced before anything else to avoid segfault due to conflict
    # with TensorFlow libraries.  Use `pa` package reference to ensure it's loaded before
    # functions like `deserialize_model` which are implemented at the top level.
    # See https://jira.apache.org/jira/browse/ARROW-3346
    pa

    import atexit
    import horovod.tensorflow.keras as hvd
    from horovod.spark.task import get_available_devices
    import os
    import tempfile
    import tensorflow as tf
    import tensorflow.keras.backend as K
    import shutil

    gpus = get_available_devices()
    if gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus[0]
    if args.dataloader == NVTABULAR_DATALOADER:
        os.environ['TF_MEMORY_ALLOCATION'] = '0.85'
        from nvtabular.loader.tensorflow import KerasSequenceLoader

    # Horovod: initialize Horovod inside the trainer.
    hvd.init()

    # Horovod: restore from checkpoint, use hvd.load_model under the hood.
    model = build_model(dimensions, args)

    # Horovod: adjust learning rate based on number of processes.
    scaled_lr = K.get_value(model.optimizer.lr) * hvd.size()
    K.set_value(model.optimizer.lr, scaled_lr)

    # Horovod: print summary logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard, or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=5, verbose=verbose),

        # Reduce LR if the metric is not improved for 10 epochs, and stop training
        # if it has not improved for 20 epochs.
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', patience=10, verbose=verbose),
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='min', patience=20, verbose=verbose),
        tf.keras.callbacks.TerminateOnNaN(),

        # Log Tensorboard events.
        tf.keras.callbacks.TensorBoard(log_dir=args.logs_dir, write_steps_per_second=True, update_freq=10)
    ]

    # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        ckpt_dir = tempfile.mkdtemp()
        ckpt_file = os.path.join(ckpt_dir, 'checkpoint.h5')
        atexit.register(lambda: shutil.rmtree(ckpt_dir))
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            ckpt_file, monitor='val_auc', mode='min', save_best_only=True))

    if args.dataloader == PETASTORM_DATALOADER:
        from petastorm import make_batch_reader
        from petastorm.tf_utils import make_petastorm_dataset

        # Make Petastorm readers.
        with make_batch_reader(f'{args.data_dir}/train',
                               num_epochs=None,
                               cur_shard=hvd.rank(),
                               shard_count=hvd.size(),
                               hdfs_driver='libhdfs') as train_reader:
            with make_batch_reader(f'{args.data_dir}/val',
                                   num_epochs=None,
                                   cur_shard=hvd.rank(),
                                   shard_count=hvd.size(),
                                   hdfs_driver='libhdfs') as val_reader:
                # Convert readers to tf.data.Dataset.
                train_ds = make_petastorm_dataset(train_reader) \
                    .unbatch() \
                    .shuffle(10 * args.batch_size) \
                    .batch(args.batch_size) \
                    .map(lambda x: (tuple(getattr(x, c) for c in ALL_COLUMNS), x.clicked))

                val_ds = make_petastorm_dataset(val_reader) \
                    .unbatch() \
                    .batch(args.batch_size) \
                    .map(lambda x: (tuple(getattr(x, c) for c in ALL_COLUMNS), x.clicked))

                history = model.fit(train_ds,
                                    validation_data=val_ds,
                                    steps_per_epoch=int(train_rows / args.batch_size / hvd.size()),
                                    validation_steps=int(val_rows / args.batch_size / hvd.size()),
                                    callbacks=callbacks,
                                    verbose=verbose,
                                    epochs=args.epochs)

    else:
        import cupy

        def seed_fn():
            """
            Generate consistent dataloader shuffle seeds across workers
            Reseeds each worker's dataloader each epoch to get fresh a shuffle
            that's consistent across workers.
            """
            min_int, max_int = tf.int32.limits
            max_rand = max_int // hvd.size()
            # Generate a seed fragment on each worker
            seed_fragment = cupy.random.randint(0, max_rand).get()
            # Aggregate seed fragments from all Horovod workers
            seed_tensor = tf.constant(seed_fragment)
            reduced_seed = hvd.allreduce(seed_tensor, name="shuffle_seed", op=hvd.Sum)
            return reduced_seed % max_rand

        train_ds = KerasSequenceLoader(
            f'{args.data_dir}/train',
            batch_size=args.batch_size,
            label_names=LABEL_COLUMNS,
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            engine="parquet",
            shuffle=True,
            buffer_size=0.06,  # how many batches to load at once
            parts_per_chunk=1,
            global_size=hvd.size(),
            global_rank=hvd.rank(),
            seed_fn=seed_fn)

        val_ds = KerasSequenceLoader(
            f'{args.data_dir}/val',
            batch_size=args.batch_size,
            label_names=LABEL_COLUMNS,
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            engine="parquet",
            shuffle=False,
            buffer_size=0.06,  # how many batches to load at once
            parts_per_chunk=1,
            global_size=hvd.size(),
            global_rank=hvd.rank())

        history = model.fit(train_ds,
                            validation_data=val_ds,
                            steps_per_epoch=int(train_rows / args.batch_size / hvd.size()),
                            validation_steps=int(val_rows / args.batch_size / hvd.size()),
                            callbacks=callbacks,
                            verbose=verbose,
                            epochs=args.epochs)

    if hvd.rank() == 0:
        return history.history


def train(dimensions, train_rows, val_rows, args):
    # Horovod: run training.
    history = horovod.spark.run(train_fn,
                                args=(dimensions, train_rows, val_rows, args),
                                num_proc=args.num_proc,
                                extra_mpi_args='-mca btl_tcp_if_include enp134s0f0 -x NCCL_IB_GID_INDEX=3',
                                stdout=sys.stdout,
                                stderr=sys.stderr,
                                verbose=2,
                                nics={},
                                prefix_output_with_timestamp=True)[0]

    best_val_loss = min(history['val_loss'])
    print('Best Loss: %f' % best_val_loss)


def main():
    parser = argparse.ArgumentParser(description='Criteo Spark Keras Training Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', default='file:///opt/data/criteo/parquet',
                        help='location of the transformed Criteo dataset in Parquet format')
    parser.add_argument('--logs-dir', default='/opt/experiments/criteo', help='location of TensorFlow logs')
    parser.add_argument('--dataloader', default=PETASTORM_DATALOADER,
                        choices=[PETASTORM_DATALOADER, NVTABULAR_DATALOADER],
                        help='dataloader to use')
    parser.add_argument('--num-proc', type=int, default=1, help='number of worker processes for training')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=64 * 1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train')
    parser.add_argument('--local-checkpoint-file', default='checkpoint', help='model checkpoint')
    args = parser.parse_args()

    spark = SparkSession.builder.appName('Criteo Keras Training').getOrCreate()

    dimensions = get_category_dimensions(spark, args.data_dir)

    train_df = spark.read.parquet(f'{args.data_dir}/train')
    val_df = spark.read.parquet(f'{args.data_dir}/val')
    test_df = spark.read.parquet(f'{args.data_dir}/test')
    train_rows, val_rows, test_rows = train_df.count(), val_df.count(), test_df.count()
    print('Training: %d' % train_rows)
    print('Validation: %d' % val_rows)
    print('Test: %d' % test_rows)

    train(dimensions, train_rows, val_rows, args)

    spark.stop()


if __name__ == '__main__':
    main()
