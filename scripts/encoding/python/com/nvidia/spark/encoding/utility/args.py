#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
import sys

from argparse import ArgumentParser
from distutils.util import strtobool

def _to_bool(literal):
    return bool(strtobool(literal))

def _to_str_list(literal):
    return [x for x in literal.split(',') if x]

_examples = [
    'com.nvidia.spark.encoding.criteo.one_hot_cpu_main',
    'com.nvidia.spark.encoding.criteo.target_cpu_main'
]

def _validate_args(args):
    usage = ''
    if args.mode == 'transform' and not args.outputPaths:
        usage += '  --outputPaths required for transform.\n'
    # for production:
    #     validates that --columns and --inputPaths exists
    #     validates that --inputPath and --outputPath matches for transform
    if (args.mainClass == 'com.nvidia.spark.encoding.criteo.target_cpu_main'
            and args.mode == 'train'
            and not args.labelColumn):
        usage += '  --labelColumn required for target encoding. \n'
    if usage:
        print('-' * 80)
        print('Usage:\n' + usage)
        sys.exit(1)

def parse_arguments():
    parser = ArgumentParser()

    # application arguments
    parser.add_argument('--mainClass', required=True, choices=_examples)
    parser.add_argument('--mode', choices=['train', 'transform'], required=True)
    parser.add_argument('--format', choices=['csv'], default='csv')
    parser.add_argument('--columns', type=_to_str_list, required=True)
    parser.add_argument('--modelPaths', type=_to_str_list, required=True)
    parser.add_argument('--inputPaths', type=_to_str_list, required=True)
    parser.add_argument('--outputPaths', type=_to_str_list)             # for transform, required
    parser.add_argument('--overwrite', type=_to_bool, default=False)
    parser.add_argument('--numRows', type=int)                          # for transform, optional
    parser.add_argument('--labelColumn', help='name of the label column') # for target encoding, required

    parsed = parser.parse_args()
    _validate_args(parsed)

    return parsed
