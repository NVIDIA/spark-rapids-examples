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

import json
import os
import time
import traceback
from typing import Callable

from pyspark import is_remote_only
from pyspark.sql import SparkSession

class PysparkBenchReport:
    """Class to generate json summary report for a benchmark
    """
    def __init__(self, spark_session: SparkSession) -> None:
        self.spark_session = spark_session
        self.summary = {
            'env': {
                'envVars': {},
                'sparkConf': {},
                'sparkVersion': None
            },
            'queryStatus': [],
            'exceptions': [],
            'startTime': None,
            'queryTimes': [],
        }

    def report_on(self, fn: Callable, *args):
        """Record a function for its running environment, running status etc. and exclude sentive
        information like tokens, secret and password Generate summary in dict format for it.

        Args:
            fn (Callable): a function to be recorded

        Returns:
            dict: summary of the fn
        """
        if not is_remote_only():
            spark_conf = dict(self.spark_session.sparkContext._conf.getAll())
        else:
            spark_conf = dict(self.spark_session.conf.getAll)
        env_vars = dict(os.environ)
        redacted = ["TOKEN", "SECRET", "PASSWORD"]
        filtered_env_vars = dict((k, env_vars[k]) for k in env_vars.keys() if not (k in redacted))
        self.summary['env']['envVars'] = filtered_env_vars
        self.summary['env']['sparkConf'] = spark_conf
        self.summary['env']['sparkVersion'] = self.spark_session.version
        listener = None

        if not is_remote_only():
            try:
                listener = python_listener.PythonListener()
                listener.register()
            except TypeError as e:
                print("Not found com.nvidia.spark.rapids.listener.Manager", str(e))
                listener = None
        if listener is not None:
            print("TaskFailureListener is registered.")
        try:
            start_time = int(time.time() * 1000)
            fn(*args)
            end_time = int(time.time() * 1000)
            if listener and len(listener.failures) != 0:
                self.summary['queryStatus'].append("CompletedWithTaskFailures")
            else:
                self.summary['queryStatus'].append("Completed")
        except Exception as e:
            # print the exception to ease debugging
            print('ERROR BEGIN')
            print(e)
            traceback.print_tb(e.__traceback__)
            print('ERROR END')
            end_time = int(time.time() * 1000)
            self.summary['queryStatus'].append("Failed")
            self.summary['exceptions'].append(str(e))
        finally:
            self.summary['startTime'] = start_time
            self.summary['queryTimes'].append(end_time - start_time)
            if listener is not None:
                listener.unregister()
            return self.summary

    def write_summary(self, query_name, prefix=""):
        """_summary_

        Args:
            query_name (str): name of the query
            prefix (str, optional): prefix for the output json summary file. Defaults to "".
        """
        # Power BI side is retrieving some information from the summary file name, so keep this file
        # name format for pipeline compatibility
        self.summary['query'] = query_name
        filename = prefix + '-' + query_name + '-' +str(self.summary['startTime']) + '.json'
        self.summary['filename'] = filename
        with open(filename, "w") as f:
            json.dump(self.summary, f, indent=2)
