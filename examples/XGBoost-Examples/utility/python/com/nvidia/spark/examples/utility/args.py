#
# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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
from argparse import ArgumentParser
from distutils.util import strtobool
from re import match
from sys import exit

def _to_bool(literal):
    return bool(strtobool(literal))

def _to_ratio_pair(literal):        # e.g., '80:20'
    return match(r'^\d+:\d+$', literal) and [int(x) for x in literal.split(':')]

MAX_CHUNK_SIZE = 2 ** 31 - 1

_examples = [
    'com.nvidia.spark.examples.agaricus.cpu_main',
    'com.nvidia.spark.examples.agaricus.gpu_main',
    'com.nvidia.spark.examples.mortgage.cpu_main',
    'com.nvidia.spark.examples.mortgage.gpu_main',
    'com.nvidia.spark.examples.mortgage.gpu_cross_validator_main',
    'com.nvidia.spark.examples.mortgage.cpu_cross_validator_main',
    'com.nvidia.spark.examples.taxi.cpu_main',
    'com.nvidia.spark.examples.taxi.gpu_main',
    'com.nvidia.spark.examples.taxi.gpu_cross_validator_main',
    'com.nvidia.spark.examples.taxi.cpu_cross_validator_main',
    'com.nvidia.spark.examples.mortgage.etl_main',
    'com.nvidia.spark.examples.taxi.etl_main'
]

_xgboost_simple_args = [
    ('cacheTrainingSet', _to_bool),
    ('maximizeEvaluationMetrics', _to_bool),
    ('useExternalMemory', _to_bool),
    ('checkpointInterval', int),
    ('maxBins', int),
    ('maxDepth', int),
    ('maxLeaves', int),
    ('nthread', int),
    ('numClass', int),
    ('numEarlyStoppingRounds', int),
    ('numRound', int),
    ('numWorkers', int),
    ('seed', int),
    ('silent', int),
    ('timeoutRequestWorkers', int),
    ('treeLimit', int),
    ('verbosity', int),
    ('alpha', float),
    ('baseScore', float),
    ('colsampleBylevel', float),
    ('colsampleBytree', float),
    ('eta', float),
    ('gamma', float),
    ('lambda_', float),
    ('lambdaBias', float),
    ('maxDeltaStep', float),
    ('minChildWeight', float),
    ('missing', float),
    ('rateDrop', float),
    ('scalePosWeight', float),
    ('sketchEps', float),
    ('skipDrop', float),
    ('subsample', float),
    ('trainTestRatio', float),
    ('baseMarginCol', str),
    ('checkpointPath', str),
    ('contribPredictionCol', str),
    ('evalMetric', str),
    ('featuresCol', str),
    ('groupCol', str),
    ('growPolicy', str),
    ('interactionConstraints', str),
    ('labelCol', str),
    ('leafPredictionCol', str),
    ('monotoneConstraints', str),
    ('normalizeType', str),
    ('objective', str),
    ('objectiveType', str),
    ('predictionCol', str),
    ('probabilityCol', str),
    ('rawPredictionCol', str),
    ('sampleType', str),
    ('treeMethod', str),
    ('weightCol', str),
]

_xgboost_array_args = [
    ('thresholds', float),
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

    # xgboost simple args
    for arg, arg_type in _xgboost_simple_args:
        parser.add_argument('--' + arg, type=arg_type)

    # xgboost array args
    for arg, arg_type in _xgboost_array_args:
        parser.add_argument('--' + arg, type=arg_type, action='append')

    parsed_all = parser.parse_args()
    _validate_args(parsed_all)
    _attach_derived_args(parsed_all)

    xgboost_args = [ arg for (arg, _) in _xgboost_simple_args + _xgboost_array_args ]
    parsed_xgboost = {
        k: v
        for k, v in vars(parsed_all).items()
        if k in xgboost_args and v is not None
    }

    return parsed_all, parsed_xgboost
