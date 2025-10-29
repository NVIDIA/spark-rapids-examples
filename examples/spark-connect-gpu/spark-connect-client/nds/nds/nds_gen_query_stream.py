#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import subprocess
import sys

from .check import check_build, check_version, get_abs_path

check_version()

def generate_query_streams(args, tool_path):
    """call TPC-DS dsqgen tool to generate a specific query or query stream(s) that contains all
    TPC-DS queries.

    Args:
        args (Namespace): Namespace from argparser
        tool_path (str): path to the tool
    """
    # move to the tools directory
    work_dir = tool_path.parent
    output_dir = get_abs_path(args.output_dir)
    template_dir = get_abs_path(args.template_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    base_cmd = ['./dsqgen',
                '-scale', args.scale,
                '-directory', template_dir,
                '-dialect', 'spark',
                '-output_dir', output_dir]
    
    if args.streams:
        cmd = base_cmd + ['-input', template_dir + '/' + 'templates.lst',
                          '-streams', args.streams]
    else:
        cmd = base_cmd + ['-template', args.template]
    if args.rngseed:
        cmd += ['-rngseed', args.rngseed]
    subprocess.run(cmd, check=True, cwd=str(work_dir))

    if args.template:
        # It's specific query, rename the stream file to its template query name
        # Special cases for query 14,23,24,39. They contains two queries in one template
        if any(q_num in args.template for q_num in ['14', '23', '24', '39']):
            with open(output_dir + '/' + 'query_0.sql', 'r') as f:
                full_content = f.read()
                part_1, part_2 = split_special_query(full_content)
            with open(output_dir + '/' + args.template[:-4] + '_part1.sql', 'w') as f:
                f.write(part_1)
            with open(output_dir + '/' + args.template[:-4] + '_part2.sql', 'w') as f:
                f.write(part_2)
            cmd = ['rm',  output_dir + '/' + 'query_0.sql']
            subprocess.run(cmd, check=True, cwd=str(work_dir))
        else:
            subprocess.run(['mv',
                            output_dir + '/' + 'query_0.sql',
                            output_dir + '/' + args.template[:-4] + '.sql'],
                           check=True, cwd=str(work_dir))

def split_special_query(q):
    split_q = q.split(';')
    # now split_q has 3 items:
    # 1. "query x in stream x using template query[xx].tpl query_part_1"
    # 2. "query_part_2"
    # 3. "-- end query [x] in stream [x] using template query[xx].tpl"
    part_1 = split_q[0].replace('.tpl', '_part1.tpl')
    part_1 += ';'
    head = split_q[0].split('\n')[0]
    part_2 = head.replace('.tpl', '_part2.tpl') + '\n'
    part_2 += split_q[1]
    part_2 += ';'
    return part_1, part_2

if __name__ == "__main__":
    _, tool_path = check_build()
    parser = parser = argparse.ArgumentParser()
    parser.add_argument('template_dir',
                        help='directory to find query templates and dialect file.')
    parser.add_argument("scale",
                        help="assume a database of this scale factor."
    )
    parser.add_argument("output_dir",
                        help="generate query in directory.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--template",
                        help="build queries from this template. Only used to generate one query " +
                        "from one tempalte. This argument is mutually exclusive with --streams. " +
                        "It is often used for test purpose.")
    group.add_argument('--streams',
                        help='generate how many query streams. ' +
                        'This argument is mutually exclusive with --template.')
    parser.add_argument('--rngseed',
                        help='seed the random generation seed.')


    args = parser.parse_args()

    generate_query_streams(args, tool_path) 
