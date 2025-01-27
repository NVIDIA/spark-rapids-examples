#
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import inspect
import socket
import psutil
import signal
import time
from pyspark import RDD
from multiprocessing import Process
from pytriton.client import ModelClient
from typing import Dict, List, Optional


def start_triton(triton_server_fn: callable, ports: List[int], model_name: str, model_path: Optional[str] = None) -> List[tuple]:
    """
    Start a Triton server in a separate process and wait for it to be ready.
    Return the (hostname, PID) of the node and process.
    """
    sig = inspect.signature(triton_server_fn)
    params = sig.parameters

    if model_path is not None:
        assert len(params) == 2, "To pass a model_path for the server to load, make sure it accepts two arguments: ports and model_path"
        args = (ports, model_path)
    else:
        assert len(params) == 1, "To start a Triton server, the function must accept one argument: ports"
        args = (ports,)

    hostname = socket.gethostname()
    process = Process(target=triton_server_fn, args=args)
    process.start()

    client = ModelClient(f"http://localhost:{ports[0]}", model_name)
    patience = 10
    while patience > 0:
        try:
            client.wait_for_model(5)
            return [(hostname, process.pid)]
        except Exception:
            print("Waiting for server to be ready...")
            patience -= 1

    emsg = "Failure: client waited too long for server startup. Check the executor logs for more info."
    raise TimeoutError(emsg)

def find_ports() -> List[int]:
    """
    Find three available network ports starting from port 7000 for Triton's HTTP, gRPC, and metrics services.
    """
    ports = []
    conns = {conn.laddr.port for conn in psutil.net_connections(kind="inet")}
    i = 7000
    while len(ports) < 3:
        if i not in conns:
            ports.append(i)
        i += 1
    
    return ports

def use_stage_level_scheduling(spark, rdd: RDD) -> RDD:
    """
    From https://github.com/NVIDIA/spark-rapids-ml/blob/main/python/src/spark_rapids_ml/core.py
    Used to ensure each Triton server instance requires a full GPU to create a 1:1 executor-server mapping.
    """

    executor_cores = spark.conf.get("spark.executor.cores")
    assert executor_cores is not None, "spark.executor.cores is not set"
    executor_gpus = spark.conf.get("spark.executor.resource.gpu.amount")
    assert executor_gpus is not None and int(executor_gpus) == 1, "spark.executor.resource.gpu.amount must be set and = 1"

    from pyspark.resource.profile import ResourceProfileBuilder
    from pyspark.resource.requests import TaskResourceRequests

    # each training task requires cpu cores > total executor cores/2 which can
    # ensure each training task be sent to different executor.
    #
    # Please note that we can't set task_cores to the value which is smaller than total executor cores/2
    # because only task_gpus can't ensure the tasks be sent to different executor even task_gpus=1.0
    #
    # If spark-rapids enabled. we don't allow other ETL task running alongside training task to avoid OOM
    spark_plugins = spark.conf.get("spark.plugins", " ")
    assert spark_plugins is not None
    spark_rapids_sql_enabled = spark.conf.get("spark.rapids.sql.enabled", "true")
    assert spark_rapids_sql_enabled is not None

    task_cores = (
        int(executor_cores)
        if "com.nvidia.spark.SQLPlugin" in spark_plugins
        and "true" == spark_rapids_sql_enabled.lower()
        else (int(executor_cores) // 2) + 1
    )
    # task_gpus means how many slots per gpu address the task requires,
    # it does mean how many gpus it would like to require, so it can be any value of (0, 0.5] or 1.
    task_gpus = 1.0
    treqs = TaskResourceRequests().cpus(task_cores).resource("gpu", task_gpus)
    rp = ResourceProfileBuilder().require(treqs).build
    print(f"Reqesting stage-level resources: (cores={task_cores}, gpu={task_gpus})")

    return rdd.withResources(rp)

def stop_triton(pids: Dict[str, int]) -> List[bool]:
    """
    Stop Triton server instances by sending a SIGTERM signal.
    """
    hostname = socket.gethostname()
    pid = pids.get(hostname, None)
    assert pid is not None, f"Could not find pid for {hostname}"
    
    for _ in range(5):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return [True]
        time.sleep(5)

    return [False]