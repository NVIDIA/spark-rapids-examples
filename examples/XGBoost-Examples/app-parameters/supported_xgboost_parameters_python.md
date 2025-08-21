Supported Parameters
============================

This is a description of all the parameters available when you are running examples in this repo:

1. All [xgboost parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) are supported.
   * Please use the `camelCase`, e.g., `--treeMethod=hist`.
   * `lambda` is replaced with `lambda_`, because `lambda` is a keyword in Python.
2. `--mainClass=[app class]`: The entry class of the application to be started. Available value is one of the below classes.
   * com.nvidia.spark.examples.agaricus.main
   * com.nvidia.spark.examples.agaricus.main
   * com.nvidia.spark.examples.mortgage.main
   * com.nvidia.spark.examples.mortgage.main
   * com.nvidia.spark.examples.taxi.main
   * com.nvidia.spark.examples.taxi.main
   * com.nvidia.spark.examples.mortgage.etl_main
   * com.nvidia.spark.examples.taxi.etl_main
3. `--format=[csv|parquet|orc]`: The format of the data for training/transforming, now only supports 'csv', 'parquet' and 'orc'. *Required*.
4. `--mode=[all|train|transform]`. The behavior of the XGBoost application (meaning CPUMain and GPUMain), default is 'all' if not specified.
   * all: Do both training and transforming, will save model to 'modelPath' if specified
   * train: Do training only, will save model to 'modelPath' if specified.
   * transform: Do transforming only, 'modelPath' is required to locate the model data to be loaded.
5. `--dataPath=[prefix]::[path]`: Path to input data file(s), or path to output data files. Use it repeatly to specify multiple data paths.
   * `--dataPath=train::[path]`: Path to the training data file(s), required when mode is NOT 'transform'.
   * `--dataPath=trans::[path]`: Path to the transforming data file(s), required when mode is NOT 'train'.
   * `--dataPath=eval::[path]`: Path to the evaluation data file(s) for training. Optional.
   * `--dataPath=rawTrain::[path]`: Path to the raw data files for training, only used by taxi/CPUMain, taxi/GPUMain now to support E2E train.
   * `--dataPath=rawTrans::[path]`: Path to the raw data files for transforming, only used by taxi/CPUMain, taxi/GPUMain now to support E2E tranformation.
   * `--dataPath=rawEval::[path]`: Path to the raw data files being used as evaluation data for training. Optional.
   * `--dataPath=raw::[path]`: Path to the raw data files to be transformed by taxi/ETLMain.
   * `--dataPath=perf::[path]`,`-dataPath=acq::[path]`: Paths to the raw data files to be transformed by mortgage/ETLMain.
   * `--dataPath=out::`: Path where to place the output data files for both mortgage/ETLMain and taxi/ETLMain.
   * `--dataPath=tmp::`: Path where to place the output data files for converting raw csv format to parquet.
6. `--modelPath=[path]`: Path to save model after training, or where to load model for transforming only. Required only when mode is 'transform'.
7. `--overwrite=[true|false]`: Whether to overwrite the current model data under 'modelPath'. Default is false. You may need to set to true to avoid IOException when saving the model to a path already exists.
8. `--hasHeader=[true|false]`: Indicate whether the csv file has header.
9. `--numRows=[int value]`: The number of the rows to be shown after transforming done. Default is 5.
10. `--showFeatures=[true|false]`: Whether to show the features columns after transforming done. Default is true.
11. `--dataRatios=[trainRatio:transformRatio]`: The ratios of data for train and transform, then the ratio for evaluation is (100-train-test). Default is 80:20, no evaluation. This is only used by taxi/ETLMain now to generate the output data.
