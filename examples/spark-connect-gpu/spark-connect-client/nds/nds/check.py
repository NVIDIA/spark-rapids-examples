#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
# -----
#
# Certain portions of the contents of this file are derived from TPC-DS version 3.2.0
# (retrieved from www.tpc.org/tpc_documents_current_versions/current_specifications5.asp).
# Such portions are subject to copyrights held by Transaction Processing Performance Council (“TPC”)
# and licensed under the TPC EULA (a copy of which accompanies this file as “TPC EULA” and is also
# available at http://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp) (the “TPC EULA”).
#
# You may not use this file except in compliance with the TPC EULA.
# DISCLAIMER: Portions of this file is derived from the TPC-DS Benchmark and as such any results
# obtained using this file are not comparable to published TPC-DS Benchmark results, as the results
# obtained from using this file do not comply with the TPC-DS Benchmark.
#

import argparse
import os
import sys
from pathlib import Path


def check_version():
    req_ver = (3, 6)
    cur_ver = sys.version_info
    if cur_ver < req_ver:
        raise Exception('Minimum required Python version is 3.6, but current python version is {}.'
                        .format(str(cur_ver.major) + '.' + str(cur_ver.minor)) +
                        ' Please use proper Python version')


def check_build():
    """check jar and tpcds executable

    Raises:
        Exception: the build is not done or broken

    Returns:
        PosixPath, PosixPath: path of jar and dsdgen executable
    """
    # Check if necessary executable or jars are built.
    # we assume user won't move this script.
    src_dir = Path(__file__).parent.absolute()
    jar_path = list(
        Path(src_dir / 'tpcds-gen/target').rglob("tpcds-gen-*.jar"))
    tool_path = list(Path(src_dir / 'tpcds-gen/target/tools').rglob("dsdgen"))
    if jar_path == [] or tool_path == []:
        raise Exception('Target jar file is not found in `target` folder or dsdgen executable is ' +
                        'not found in `target/tools` folder.' +
                        'Please refer to README document and build this project first.')
    return jar_path[0], tool_path[0]


def get_abs_path(input_path):
    """receive a user input path and return absolute path of it.

    Args:
        input_path (str): user's input path

    Returns:
        str: if the input is absolute, return it; if it's relative path, return the absolute path of
        it.
    """
    if Path(input_path).is_absolute():
        # it's absolute path
        output_path = input_path
    else:
        # it's relative path where this script is executed
        output_path = os.getcwd() + '/' + input_path
    return output_path


def valid_range(range, parallel):
    """check the range validation

    Args:
        range (str): a range specified for a range data generation, e.g. "1,10"
        parallel (str): string type number for parallelism in TPC-DS data generation, e.g. "20"

    Raises:
        Exception: error message for invalid range input.
    """
    if len(range.split(',')) != 2:
        msg = 'Invalid range: please specify a range with a comma between start and end. e.g., "1,10".'
        raise Exception(msg)
    range_start = int(range.split(',')[0])
    range_end = int(range.split(',')[1])
    if range_start < 1 or range_start > range_end or range_end > int(parallel):
        msg = 'Please provide correct child range: 1 <= range_start <= range_end <= parallel'
        raise Exception(msg)
    return range_start, range_end


def parallel_value_type(p):
    """helper function to check parallel valuie

    Args:
        p (str): parallel value

    Raises:
        argparse.ArgumentTypeError: ArgumentTypeError exception

    Returns:
        str: parallel in string
    """
    if int(p) < 2:
        raise argparse.ArgumentTypeError("PARALLEL must be >= 2")
    return p


def get_dir_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def check_json_summary_folder(json_summary_folder):
    if json_summary_folder:
    # prepare a folder to save json summaries of query results
        if not os.path.exists(json_summary_folder):
            os.makedirs(json_summary_folder)
        else:
            if os.listdir(json_summary_folder):
                raise Exception(f"json_summary_folder {json_summary_folder} is not empty. " +
                                "There may be already some json files there. Please clean the folder " +
                                "or specify another one.")

def check_query_subset_exists(query_dict, subset_list):
    """check if the query subset exists in the query dictionary"""
    for q in subset_list:
        if q not in query_dict.keys():
            raise Exception(f"Query {q} is not in the query dictionary. Please check the query subset.")
    return True
