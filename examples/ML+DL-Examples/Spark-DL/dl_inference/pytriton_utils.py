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
import inspect
import logging
import os
import signal
import socket
import time
from multiprocessing import Process
from typing import Dict, List, Optional

import psutil
from pyspark import RDD
from pyspark.sql import SparkSession

from pytriton.client import ModelClient

DEFAULT_WAIT_RETRIES = 10
DEFAULT_WAIT_TIMEOUT = 5

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TritonManager")


def _start_triton_server(
    triton_server_fn: callable,
    ports: List[int],
    model_name: str,
    model_path: Optional[str] = None,
    max_retries: int = DEFAULT_WAIT_RETRIES,
    wait_timeout: int = DEFAULT_WAIT_TIMEOUT,
) -> List[tuple]:
    """Task to start Triton server process on a Spark executor."""
    sig = inspect.signature(triton_server_fn)
    params = sig.parameters

    if model_path is not None:
        assert (
            len(params) == 2
        ), "Server function must accept (ports, model_path) when model_path is provided"
        args = (ports, model_path)
    else:
        assert len(params) == 1, "Server function must accept (ports) argument"
        args = (ports,)

    hostname = socket.gethostname()
    process = Process(target=triton_server_fn, args=args)
    process.start()

    client = ModelClient(f"http://localhost:{ports[0]}", model_name)

    for _ in range(max_retries):
        try:
            client.wait_for_model(wait_timeout)
            return [(hostname, process.pid)]
        except Exception:
            print("Waiting for server to be ready...")

    raise TimeoutError(
        "Failure: server startup timeout exceeded. Check the executor logs for more info."
    )


def _stop_triton_server(
    server_pids: Dict[str, int],
    max_retries: int = DEFAULT_WAIT_RETRIES,
    retry_delay: int = DEFAULT_WAIT_TIMEOUT,
) -> List[bool]:
    """Task to stop the Triton server on a Spark executor."""
    hostname = socket.gethostname()
    pid = server_pids.get(hostname)
    assert pid is not None, f"No server PID found for host {hostname}"

    for _ in range(max_retries):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return [True]
        time.sleep(retry_delay)

    return [False]  # Failed to terminate or timed out


class TritonServerManager:
    """
    Handle lifecycle of Triton server instances across Spark cluster, e.g.
    - Find available ports
    - Start server processes across executors via stage-level scheduling
    - Gracefully shutdown servers across executors

    Attributes:
        spark: Active SparkSession
        num_nodes: Number of Triton servers to manage (= # of executors/GPUs)
        model_name: Name of the model being served
        model_path: Optional path to model files
        server_pids: Dictionary of hostname to server process IDs
        ports: List of ports used for HTTP, gRPC, and metrics services

    Example usage (4 node cluster):
    >>> server_manager = TritonServerManager(num_nodes=4, model_name="my_model", model_path="/path/to/my_model")
    >>> # Define triton_server_fn(ports, model_path) that contains PyTriton server logic
    >>> server_pids = server_manager.start_servers(triton_server_fn)
    >>> print(f"Servers started with PIDs: {server_pids}")
    >>> http_url = server_manager.http_url
    >>> grpc_url = server_manager.grpc_url
    >>> # Run inference...
    >>> success = server_manager.stop_servers()
    >>> print(f"Server shutdown success: {success}")
    """

    def __init__(
        self, num_nodes: int, model_name: str, model_path: Optional[str] = None
    ):
        """
        Initialize the Triton server manager.

        Args:
            num_nodes: Number of executors (GPUs) in cluster
            model_name: Name of the model to serve
            model_path: Optional path to model file for server function to load from disk
        """
        self.spark = SparkSession.getActiveSession()
        self.num_nodes = num_nodes
        self.model_name = model_name
        self.model_path = model_path
        self._server_pids: Dict[str, int] = {}
        self._ports: Optional[List[int]] = []

    @property
    def http_url(self):
        """Returns client HTTP URL if ports are available (servers are running)."""
        if not self._ports:
            return None
        return f"http://localhost:{self._ports[0]}"

    @property
    def grpc_url(self):
        """Returns client gRPC URL if ports are available (servers are running)."""
        if not self._ports:
            return None
        return f"grpc://localhost:{self._ports[1]}"

    def _find_ports(self, start_port: int = 7000) -> List[int]:
        """Find available ports for Triton's HTTP, gRPC, and metrics services."""

        def _find_ports_task(start_port: int) -> List[int]:
            """Task to find ports on the Spark worker."""
            ports = []
            conns = {conn.laddr.port for conn in psutil.net_connections(kind="inet")}
            i = start_port

            while len(ports) < 3:
                if i not in conns:
                    ports.append(i)
                i += 1

            return ports

        # For simplicity assume that ports available on one worker are available on all workers.
        rdd = self.spark.sparkContext.parallelize([start_port], numSlices=1)
        return rdd.map(lambda start_port: _find_ports_task(start_port)).collect()[0]

    def _get_node_rdd(self) -> RDD:
        """Create and configure RDD with stage-level scheduling for 1 task per node."""
        sc = self.spark.sparkContext
        node_rdd = sc.parallelize(list(range(self.num_nodes)), self.num_nodes)
        return self._use_stage_level_scheduling(node_rdd)

    def _use_stage_level_scheduling(self, rdd: RDD) -> RDD:
        """
        Use stage-level scheduling to ensure each Triton server instance maps to 1 GPU (executor).
        From https://github.com/NVIDIA/spark-rapids-ml/blob/main/python/src/spark_rapids_ml/core.py
        """
        executor_cores = self.spark.conf.get("spark.executor.cores")
        assert executor_cores is not None, "spark.executor.cores is not set"
        executor_gpus = self.spark.conf.get("spark.executor.resource.gpu.amount")
        assert (
            executor_gpus is not None and int(executor_gpus) == 1
        ), "spark.executor.resource.gpu.amount must be set and = 1"

        from pyspark.resource.profile import ResourceProfileBuilder
        from pyspark.resource.requests import TaskResourceRequests

        spark_plugins = self.spark.conf.get("spark.plugins", " ")
        assert spark_plugins is not None
        spark_rapids_sql_enabled = self.spark.conf.get(
            "spark.rapids.sql.enabled", "true"
        )
        assert spark_rapids_sql_enabled is not None

        task_cores = (
            int(executor_cores)
            if "com.nvidia.spark.SQLPlugin" in spark_plugins
            and "true" == spark_rapids_sql_enabled.lower()
            else (int(executor_cores) // 2) + 1
        )
        task_gpus = 1.0
        treqs = TaskResourceRequests().cpus(task_cores).resource("gpu", task_gpus)
        rp = ResourceProfileBuilder().require(treqs).build
        logger.info(
            f"Requesting stage-level resources: (cores={task_cores}, gpu={task_gpus})"
        )

        return rdd.withResources(rp)

    def start_servers(self, triton_server_fn: callable) -> Dict[str, int]:
        """
        Start Triton servers across the cluster.

        Args:
            triton_server_fn: PyTriton server function defining the model and inference logic

        Returns:
            Dictionary of hostname -> server PID
        """
        node_rdd = self._get_node_rdd()
        ports = self._find_ports()
        model_name = self.model_name
        model_path = self.model_path

        logger.info(
            f"Starting {self.num_nodes} servers using ports {ports} for HTTP, gRPC, and metrics."
        )

        self._server_pids = (
            node_rdd.barrier()
            .mapPartitions(
                lambda _: _start_triton_server(
                    triton_server_fn=triton_server_fn,
                    ports=ports,
                    model_name=model_name,
                    model_path=model_path,
                )
            )
            .collectAsMap()
        )

        self._ports = ports

        return self._server_pids

    def stop_servers(self) -> List[bool]:
        """
        Stop all Triton servers across the cluster.

        Returns:
            List of booleans indicating success/failure of stopping each server
        """
        if not self._server_pids:
            logger.warning("No servers to stop.")
            return

        node_rdd = self._get_node_rdd()
        server_pids = self._server_pids

        stop_success = (
            node_rdd.barrier()
            .mapPartitions(lambda _: _stop_triton_server(server_pids))
            .collect()
        )

        if all(stop_success):
            self._server_pids.clear()
            self._ports.clear()
            logger.info(f"Sucessfully stopped {self.num_nodes} servers.")
        else:
            logger.warning(
                f"Server termination failed or timed out. Check executor logs."
            )

        return stop_success
