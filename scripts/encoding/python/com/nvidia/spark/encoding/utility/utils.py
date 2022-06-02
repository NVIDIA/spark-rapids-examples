#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pickle

def load_data(spark, paths, args, customize=None):
    reader = (spark
        .read
        .format(args.format))
    customize and customize(reader)
    return reader.load(paths)

def save_data(data_frame, path, args, customize=None):
    writer = (data_frame
        .write
        .format(args.format))
    args.overwrite and writer.mode('overwrite')
    customize and customize(writer)
    writer.save(path)

def load_model(model_class, path):
    return model_class.load(path)

def load_models(model_class, paths):
    return [load_model(model_class, path) for path in paths]

def save_model(model, path, args):
    writer = model.write().overwrite() if args.overwrite else model
    writer.save(path)

def save_dict(mean_dict, target_path):
    '''
    target_path: full path of the target location to save the dict
    '''
    with open(target_path+'.pkl', 'wb') as f:
        pickle.dump(mean_dict, f, pickle.HIGHEST_PROTOCOL)

def load_dict(dict_path):
    '''
    dict_path: full path of target dict with '.pkl' tail.
    '''
    with open(dict_path, 'rb') as f:
        return pickle.load(f)

def load_dict_df(spark, dict_df_path):
    return spark.read.option("header","false").csv(dict_df_path)
