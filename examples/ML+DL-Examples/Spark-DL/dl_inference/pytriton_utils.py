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
import sys
import time
from multiprocessing import Process
from typing import Callable, Dict, List, Optional, Tuple

import psutil
from pyspark import RDD
from pyspark.sql import SparkSession

from pytriton.client import ModelClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TritonManager")


def _start_triton_server(
    triton_server_fn: Callable,
    model_name: str,
    wait_retries: int,
    wait_timeout: int,
    model_path: Optional[str] = None,
) -> List[tuple]:
    """Task to start Triton server process on a Spark executor."""

    from pyspark import BarrierTaskContext

    def _prepare_pytriton_env():
        """Expose PyTriton to correct libpython3.11.so and Triton bundled libraries."""
        ld_library_paths = []

        # Add nvidia_pytriton.libs to LD_LIBRARY_PATH
        for path in sys.path:
            if os.path.isdir(path) and "site-packages" in path:
                libs_path = os.path.join(path, "nvidia_pytriton.libs")
                if os.path.isdir(libs_path):
                    ld_library_paths.append(libs_path)
                    break

        # Add ${CONDA_PREFIX}/lib to LD_LIBRARY_PATH for conda environments
        if os.path.exists(os.path.join(sys.prefix, "conda-meta")):
            conda_lib = os.path.join(sys.prefix, "lib")
            if os.path.isdir(conda_lib):
                ld_library_paths.append(conda_lib)

        if "LD_LIBRARY_PATH" in os.environ:
            ld_library_paths.append(os.environ["LD_LIBRARY_PATH"])

        os.environ["LD_LIBRARY_PATH"] = ":".join(ld_library_paths)

        return None

    def _find_ports(start_port: int = 7000) -> List[int]:
        """Find available ports for Triton's HTTP, gRPC, and metrics services."""
        ports = []
        conns = {conn.laddr.port for conn in psutil.net_connections(kind="inet")}
        i = start_port

        while len(ports) < 3:
            if i not in conns:
                ports.append(i)
            i += 1

        return ports

    tc = BarrierTaskContext.get()
    ports = _find_ports()
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

    _prepare_pytriton_env()
    hostname = socket.gethostname()
    process = Process(target=triton_server_fn, args=args)
    process.start()

    client = ModelClient(f"http://localhost:{ports[0]}", model_name)

    for _ in range(wait_retries):
        try:
            client.wait_for_model(wait_timeout)
            tc.barrier()
            client.close()
            return [(hostname, (process.pid, ports))]
        except Exception:
            pass

    client.close()
    if process.is_alive():
        # Terminate if timeout is exceeded to avoid dangling server processes
        process.terminate()

    raise TimeoutError(
        "Failure: server startup timeout exceeded. Check the executor logs for more info."
    )


def _stop_triton_server(
    server_pids_ports: Dict[str, Tuple[int, List[int]]],
    wait_retries: int,
    wait_timeout: int,
) -> List[bool]:
    """Task to stop the Triton server on a Spark executor."""
    hostname = socket.gethostname()
    pid, _ = server_pids_ports.get(hostname)
    assert pid is not None, f"No server PID found for host {hostname}"

    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=wait_timeout * wait_retries)
        return [True]
    except psutil.NoSuchProcess:
        return [True]
    except psutil.TimeoutExpired:
        try:
            process.kill()
            return [True]
        except:
            return [False]


class TritonServerManager:
    """
    Handle lifecycle of Triton server instances across Spark cluster, e.g.
    - Find available ports
    - Start server processes across executors via stage-level scheduling
    - Gracefully shutdown servers across executors

    Attributes:
        spark: Active SparkSession
        num_executors: Number of Triton servers to manage (= # of executors)
        model_name: Name of the model being served
        model_path: Optional path to model files
        server_pids_ports: Dictionary of hostname to (server process ID, ports)

    Example usage:
    >>> server_manager = TritonServerManager(model_name="my_model", model_path="/path/to/my_model")
    >>> # Define triton_server(ports, model_path) that contains PyTriton server logic
    >>> server_pids_ports = server_manager.start_servers(triton_server)
    >>> print(f"Servers started with PIDs/Ports: {server_pids_ports}")
    >>> host_to_http_url = server_manager.host_to_http_url
    >>> host_to_grpc_url = server_manager.host_to_grpc_url
    >>> # Define triton_fn() and predict_batch_udf(triton_fn) and run inference...
    >>> success = server_manager.stop_servers()
    >>> print(f"Server shutdown success: {success}")
    """

    DEFAULT_WAIT_RETRIES = 10
    DEFAULT_WAIT_TIMEOUT = 5

    def __init__(
        self, model_name: str, model_path: Optional[str] = None
    ):
        """
        Initialize the Triton server manager.

        Args:
            model_name: Name of the model to serve
            model_path: Optional path to model file for server function to load from disk
        """
        self.spark = SparkSession.getActiveSession()
        self.num_nodes = self._get_num_nodes()
        self.model_name = model_name
        self.model_path = model_path
        self._server_pids_ports: Dict[str, Tuple[int, List[int]]] = {}

    def _get_num_nodes(self) -> int:
        """Get the number of executors in the cluster."""
        return len([executor.host() for executor in self.spark._jsc.sc().statusTracker().getExecutorInfos()]) - 1

    @property
    def host_to_http_url(self) -> Dict[str, str]:
        """Map hostname to client HTTP URL for Triton server on that host."""
        if not self._server_pids_ports:
            logger.warning("No urls available. Start servers first.")
            return None

        return {
            host: f"http://{host}:{ports[0]}"
            for host, (_, ports) in self._server_pids_ports.items()
        }

    @property
    def host_to_grpc_url(self) -> Dict[str, str]:
        """Map hostname to client gRPC URL for Triton server on that host."""
        if not self._server_pids_ports:
            logger.warning("No urls available. Start servers first.")
            return None

        return {
            host: f"grpc://{host}:{ports[1]}"
            for host, (_, ports) in self._server_pids_ports.items()
        }

    def _get_node_rdd(self) -> RDD:
        """Create and configure RDD with stage-level scheduling for 1 task per executor."""
        sc = self.spark.sparkContext
        node_rdd = sc.parallelize(list(range(self.num_executors)), self.num_executors)
        return self._use_stage_level_scheduling(node_rdd)

    def _use_stage_level_scheduling(self, rdd: RDD) -> RDD:
    def _use_stage_level_scheduling(self, rdd: RDD) -> RDD:
        """
        Use stage-level scheduling to ensure each Triton server instance maps to 1 executor.
        Use stage-level scheduling to ensure each Triton server instance maps to 1 executor.
        From https://github.com/NVIDIA/spark-rapids-ml/blob/main/python/src/spark_rapids_ml/core.py
        """
        from pyspark.resource.profile import ResourceProfileBuilder
        from pyspark.resource.requests import TaskResourceRequests
        
        from pyspark.resource.profile import ResourceProfileBuilder
        from pyspark.resource.requests import TaskResourceRequests
        
        executor_cores = self.spark.conf.get("spark.executor.cores")
        assert executor_cores is not None, "spark.executor.cores is not set"
        executor_gpus = self.spark.conf.get("spark.executor.resource.gpu.amount")
        assert executor_gpus is not None, "spark.executor.resource.gpu.amount is not set"
        assert executor_gpus is not None, "spark.executor.resource.gpu.amount is not set"

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
        task_gpus = float(executor_gpus)

        task_gpus = float(executor_gpus)

        treqs = TaskResourceRequests().cpus(task_cores).resource("gpu", task_gpus)
        rp = ResourceProfileBuilder().require(treqs).build
        logger.info(
            f"Requesting stage-level resources: (cores={task_cores}, gpu={task_gpus})"
        )

        return rdd.withResources(rp)

    def start_servers(
        self,
        triton_server_fn: Callable,
        wait_retries: int = DEFAULT_WAIT_RETRIES,
        wait_timeout: int = DEFAULT_WAIT_TIMEOUT,
    ) -> Dict[str, Tuple[int, List[int]]]:
        """
        Start Triton servers across the cluster.

        Args:
            triton_server_fn: PyTriton server function defining the model and inference logic

        Returns:
            Dictionary of hostname -> server PID
        """
        node_rdd = self._get_node_rdd()
        model_name = self.model_name
        model_path = self.model_path

        logger.info(f"Starting {self.num_executors} servers.")

        self._server_pids_ports = (
            node_rdd.barrier()
            .mapPartitions(
                lambda _: _start_triton_server(
                    triton_server_fn=triton_server_fn,
                    model_name=model_name,
                    wait_retries=wait_retries,
                    wait_timeout=wait_timeout,
                    model_path=model_path,
                )
            )
            .collectAsMap()
        )

        return self._server_pids_ports

    def stop_servers(
        self,
        wait_retries: int = DEFAULT_WAIT_RETRIES,
        wait_timeout: int = DEFAULT_WAIT_TIMEOUT,
    ) -> List[bool]:
        """
        Stop all Triton servers across the cluster.

        Returns:
            List of booleans indicating success/failure of stopping each server
        """
        if not self._server_pids_ports:
            logger.warning("No servers to stop.")
            return

        node_rdd = self._get_node_rdd()
        server_pids_ports = self._server_pids_ports

        stop_success = (
            node_rdd.barrier()
            .mapPartitions(
                lambda _: _stop_triton_server(
                    server_pids_ports=server_pids_ports,
                    wait_retries=wait_retries,
                    wait_timeout=wait_timeout,
                )
            )
            .collect()
        )

        if all(stop_success):
            self._server_pids_ports.clear()
            logger.info(f"Sucessfully stopped {self.num_executors} servers.")
        else:
            logger.warning(
                f"Server termination failed or timed out. Check executor logs."
            )

        return stop_success
