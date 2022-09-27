#
# Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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
import typing
from argparse import ArgumentParser
from distutils.util import strtobool
from re import match
from sys import exit


def _to_bool(literal):
    return bool(strtobool(literal))


def _to_ratio_pair(literal):  # e.g., '80:20'
    return match(r'^\d+:\d+$', literal) and [int(x) for x in literal.split(':')]


MAX_CHUNK_SIZE = 2 ** 31 - 1

_examples = [
    'com.nvidia.spark.examples.agaricus.main',
    'com.nvidia.spark.examples.mortgage.main',
    'com.nvidia.spark.examples.mortgage.etl_main',
    'com.nvidia.spark.examples.mortgage.cross_validator_main',
    'com.nvidia.spark.examples.taxi.main',
    'com.nvidia.spark.examples.taxi.etl_main',
    'com.nvidia.spark.examples.taxi.cross_validator_main',
]


def _validate_args(args):
    usage = ''
    if not args.dataPaths:
        usage += '  --dataPaths is required.\n'
    if not (args.dataRatios
            and 0 <= args.dataRatios[0] <= 100
            and 0 <= args.dataRatios[1] <= 100
            and args.dataRatios[0] + args.dataRatios[1] <= 100):
        usage += '  --dataRatios should be in format \'Int:Int\', these two ints should be' \
                 ' in range [0, 100] and the sum should be less than or equal to 100.\n'
    if not (1 <= args.maxRowsPerChunk <= MAX_CHUNK_SIZE):
        usage += '  --maxRowsPerChunk should be in range [1, {}].\n'.format(MAX_CHUNK_SIZE)
    if usage:
        print('-' * 80)
        print('Usage:\n' + usage)
        exit(1)


def _attach_derived_args(args):
    args.trainRatio = args.dataRatios[0]
    args.evalRatio = args.dataRatios[1]
    args.trainEvalRatio = 100 - args.trainRatio - args.evalRatio
    args.splitRatios = [args.trainRatio, args.trainEvalRatio, args.evalRatio]


def _inspect_xgb_parameters() -> dict[str, type]:
    """inspect XGBModel parameters from __init__"""
    from xgboost import XGBModel
    from typing import get_type_hints, get_origin
    xgb_parameters = {}
    xgb_model_sig = get_type_hints(XGBModel.__init__)
    for k, v in xgb_model_sig.items():
        if k != "kwargs" and k != "return":
            if get_origin(v) == typing.Union:
                xgb_parameters[k] = v.__args__[0]
            else:
                xgb_parameters[k] = v

    # some extra parameters used by xgboost pyspark
    xgb_parameters['objective'] = str
    xgb_parameters['force_repartition'] = _to_bool
    xgb_parameters['use_gpu'] = _to_bool
    xgb_parameters['num_workers'] = int
    xgb_parameters['enable_sparse_data_optim'] = _to_bool
    return xgb_parameters


def parse_arguments():
    parser = ArgumentParser()

    # application arguments
    parser.add_argument('--mainClass', required=True, choices=_examples)
    parser.add_argument('--mode', choices=['all', 'train', 'transform'], default='all')
    parser.add_argument('--format', required=True, choices=['csv', 'parquet', 'orc'])
    parser.add_argument('--hasHeader', type=_to_bool, default=True)
    parser.add_argument('--asFloats', type=_to_bool, default=True)
    parser.add_argument('--maxRowsPerChunk', type=int, default=MAX_CHUNK_SIZE)
    parser.add_argument('--modelPath')
    parser.add_argument('--overwrite', type=_to_bool, default=False)
    parser.add_argument('--dataPath', dest='dataPaths', action='append')
    parser.add_argument('--dataRatios', type=_to_ratio_pair, default=[80, 20])
    parser.add_argument('--numRows', type=int, default=5)
    parser.add_argument('--showFeatures', type=_to_bool, default=True)

    xgboost_all_args = _inspect_xgb_parameters()
    for arg, tp in xgboost_all_args.items():
        parser.add_argument('--' + arg, type=tp)

    parsed_all = parser.parse_args()
    _validate_args(parsed_all)
    _attach_derived_args(parsed_all)

    parsed_xgboost = {
        k: v
        for k, v in vars(parsed_all).items()
        if k in xgboost_all_args and v is not None
    }

    return parsed_all, parsed_xgboost
