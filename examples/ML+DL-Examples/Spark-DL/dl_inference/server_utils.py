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
import socket
import subprocess
import sys
import time
from multiprocessing import Process
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import psutil
import requests
from pyspark import RDD
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ServerManager")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _find_ports(num_ports: int, start_port: int = 7000) -> List[int]:
    """Find available ports on executor for server services."""
    ports = []
    conns = {conn.laddr.port for conn in psutil.net_connections(kind="inet")}
    i = start_port

    while len(ports) < num_ports:
        if i not in conns:
            ports.append(i)
        i += 1

    return ports


def _get_valid_vllm_parameters_task() -> Set[str]:
    """Task to get valid vLLM parameters on executor."""
    from vllm.entrypoints.openai.cli_args import create_parser_for_docs

    parser = create_parser_for_docs()
    valid_args = set()
    for action in parser._actions:
        if action.dest not in [
            "help",
            "host",
            "port",
            "served-model-name",
            "model",
        ]:
            valid_args.add(action.dest)

    return valid_args


def _start_triton_server_task(
    triton_server_fn: Callable,
    model_name: str,
    wait_retries: int,
    wait_timeout: int,
    model_path: Optional[str] = None,
) -> List[tuple]:
    """Task to start Triton server process on a Spark executor."""

    from pyspark import BarrierTaskContext

    from pytriton.client import ModelClient

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

    # Setup server function arguments
    tc = BarrierTaskContext.get()
    ports = _find_ports(num_ports=3)
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

    # Prepare and start server process
    _prepare_pytriton_env()
    hostname = socket.gethostname()
    process = Process(target=triton_server_fn, args=args)
    process.start()

    client = ModelClient(f"http://localhost:{ports[0]}", model_name)

    # Wait for server to start
    for _ in range(wait_retries):
        try:
            client.wait_for_model(wait_timeout)
            tc.barrier()
            client.close()
            return [(hostname, (process.pid, ports))]
        except Exception:
            if not process.is_alive():
                # If process terminated due to an error, stop waiting
                break
            pass

    client.close()
    if process.is_alive():
        # Terminate if timeout is exceeded to avoid dangling server processes
        process.terminate()

    raise TimeoutError(
        "Failure: Triton server startup failed or timed out. Check the executor logs for more info."
    )


def _start_vllm_server_task(
    model_name: str,
    model_path: str,
    wait_retries: int,
    wait_timeout: int,
    **kwargs,
) -> List[tuple]:
    """Task to start vLLM server process on a Spark executor."""
    from pyspark import BarrierTaskContext

    tc = BarrierTaskContext.get()
    port = _find_ports(num_ports=1)[0]
    hostname = socket.gethostname()

    # Build command for vLLM server
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        model_name,
        "--port",
        str(port),
    ]

    # Add additional args from kwargs
    for key, value in kwargs.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd.append(f"--{key}")
            cmd.append(str(value))

    logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")

    # vLLM does CUDA init at import time. Forking will try to re-initialize CUDA if vLLM was imported before and throw an error.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Start server process
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)

    # Wait for server to start
    health_url = f"http://localhost:{port}/health"
    for _ in range(wait_retries):
        try:
            time.sleep(wait_timeout)
            response = requests.get(health_url)
            if response.status_code == 200:
                tc.barrier()
                return [(hostname, (process.pid, [port]))]
        except Exception:
            if process.poll() is not None:
                # If process terminated due to an error, stop waiting
                break
            pass

    if process.poll() is None:
        # Terminate if timeout is exceeded to avoid dangling server processes
        process.terminate()

    raise TimeoutError(
        "Failure: vLLM server startup failed or timed out. Check the executor logs for more info."
    )


def _stop_server_task(
    server_pids_ports: Dict[str, Tuple[int, List[int]]],
    wait_retries: int,
    wait_timeout: int,
) -> List[bool]:
    """Task to stop a server process on a Spark executor."""
    hostname = socket.gethostname()
    pid, _ = server_pids_ports.get(hostname, (None, None))
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


# -----------------------------------------------------------------------------
# ServerManager Classes
# -----------------------------------------------------------------------------


class ServerManager:
    """
    Base class for server management across a Spark cluster.

    Attributes:
        spark: Active SparkSession
        num_executors: Number of servers to manage (= # of executors)
        model_name: Name of the served model
        model_path: Optional path to model files
        server_pids_ports: Dictionary of hostname to (server process ID, ports)
    """

    DEFAULT_WAIT_RETRIES = 24
    DEFAULT_WAIT_TIMEOUT = 5

    def __init__(self, model_name: str, model_path: Optional[str] = None):
        """
        Initialize the server manager.

        Args:
            model_name: Name of the model to serve
            model_path: Optional path to model file for server function to load from disk
        """
        self.spark = SparkSession.getActiveSession()
        self.num_executors = self._get_num_executors()
        self.model_name = model_name
        self.model_path = model_path
        self._server_pids_ports: Dict[str, Tuple[int, List[int]]] = {}

    def _get_num_executors(self) -> int:
        """Get the number of executors in the cluster."""
        return (
            len(
                [
                    executor.host()
                    for executor in self.spark._jsc.sc()
                    .statusTracker()
                    .getExecutorInfos()
                ]
            )
            - 1
        )

    @property
    def host_to_http_url(self) -> Dict[str, str]:
        """Map hostname to client HTTP URL for server on that host."""
        if not self._server_pids_ports:
            logger.warning("No urls available. Start servers first.")
            return None

        return {
            host: f"http://localhost:{ports[0]}"
            for host, (_, ports) in self._server_pids_ports.items()
        }

    def _get_node_rdd(self) -> RDD:
        """Create and configure RDD with stage-level scheduling for 1 task per executor."""
        sc = self.spark.sparkContext
        node_rdd = sc.parallelize(list(range(self.num_executors)), self.num_executors)
        return self._use_stage_level_scheduling(node_rdd)

    def _use_stage_level_scheduling(self, rdd: RDD) -> RDD:
        """
        Use stage-level scheduling to ensure each server instance maps to 1 executor.
        Adapted from https://github.com/NVIDIA/spark-rapids-ml/blob/main/python/src/spark_rapids_ml/core.py
        """
        from pyspark.resource.profile import ResourceProfileBuilder
        from pyspark.resource.requests import TaskResourceRequests

        executor_cores = self.spark.conf.get("spark.executor.cores")
        assert executor_cores is not None, "spark.executor.cores is not set"
        executor_gpus = self.spark.conf.get("spark.executor.resource.gpu.amount")
        assert (
            executor_gpus is not None
        ), "spark.executor.resource.gpu.amount is not set"

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

        treqs = TaskResourceRequests().cpus(task_cores).resource("gpu", task_gpus)
        rp = ResourceProfileBuilder().require(treqs).build
        logger.info(
            f"Requesting stage-level resources: (cores={task_cores}, gpu={task_gpus})"
        )

        return rdd.withResources(rp)

    def start_servers(
        self,
        start_server_fn: Callable,
        wait_retries: int = DEFAULT_WAIT_RETRIES,
        wait_timeout: int = DEFAULT_WAIT_TIMEOUT,
        **kwargs,
    ) -> Dict[str, Tuple[int, List[int]]]:
        """
        Start servers across the cluster.

        Args:
            start_server_fn: Function used to start the server process
            wait_retries: Number of retries for waiting for server startup
            wait_timeout: Timeout in seconds for each retry
            **kwargs: Additional server-specific arguments

        Returns:
            Dictionary of hostname -> (server PID, [ports])
        """
        node_rdd = self._get_node_rdd()
        model_name = self.model_name
        model_path = self.model_path
        server_type = self.__class__.__name__.replace("ServerManager", "")

        logger.info(f"Starting {self.num_executors} {server_type} servers.")

        start_args = {
            "model_name": model_name,
            "wait_retries": wait_retries,
            "wait_timeout": wait_timeout,
        }

        if model_path is not None:
            start_args["model_path"] = model_path

        start_args.update(kwargs)

        self._server_pids_ports = (
            node_rdd.barrier()
            .mapPartitions(lambda _: start_server_fn(**start_args))
            .collectAsMap()
        )

        return self._server_pids_ports

    def stop_servers(
        self,
        wait_retries: int = DEFAULT_WAIT_RETRIES,
        wait_timeout: int = DEFAULT_WAIT_TIMEOUT,
    ) -> List[bool]:
        """
        Stop all servers across the cluster.

        Returns:
            List of booleans indicating success/failure of stopping each server
        """
        if not self._server_pids_ports:
            logger.warning("No servers to stop.")
            return []

        node_rdd = self._get_node_rdd()
        server_pids_ports = self._server_pids_ports
        server_type = self.__class__.__name__.replace("ServerManager", "")

        stop_success = (
            node_rdd.barrier()
            .mapPartitions(
                lambda _: _stop_server_task(
                    server_pids_ports=server_pids_ports,
                    wait_retries=wait_retries,
                    wait_timeout=wait_timeout,
                )
            )
            .collect()
        )

        if all(stop_success):
            self._server_pids_ports.clear()
            logger.info(
                f"Successfully stopped {self.num_executors} {server_type} servers."
            )
        else:
            logger.warning(
                f"{server_type} server termination failed or timed out. Check executor logs."
            )

        return stop_success


class TritonServerManager(ServerManager):
    """
    Handle lifecycle of Triton server instances across Spark cluster.

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

    def __init__(self, model_name: str, model_path: Optional[str] = None):
        super().__init__(model_name, model_path)

    @property
    def host_to_grpc_url(self) -> Dict[str, str]:
        """Map hostname to client gRPC URL for Triton server on that host."""
        if not self._server_pids_ports:
            logger.warning("No urls available. Start servers first.")
            return None

        return {
            host: f"grpc://localhost:{ports[1]}"
            for host, (_, ports) in self._server_pids_ports.items()
        }

    def start_servers(
        self,
        triton_server_fn: Callable,
        wait_retries: int = ServerManager.DEFAULT_WAIT_RETRIES,
        wait_timeout: int = ServerManager.DEFAULT_WAIT_TIMEOUT,
    ) -> Dict[str, Tuple[int, List[int]]]:
        """
        Start Triton servers across the cluster.

        Args:
            triton_server_fn: PyTriton server function defining the model and inference logic
            wait_retries: Number of retries for waiting for server startup
            wait_timeout: Timeout in seconds for each retry

        Returns:
            Dictionary of hostname -> (server PID, [ports])
        """
        return super().start_servers(
            start_server_fn=_start_triton_server_task,
            wait_retries=wait_retries,
            wait_timeout=wait_timeout,
            triton_server_fn=triton_server_fn,
        )


class VLLMServerManager(ServerManager):
    """
    Handle lifecycle of vLLM server instances across Spark cluster.

    Example usage:
    >>> server_manager = VLLMServerManager(model_name="my_llm", model_path="/path/to/my_llm")
    >>> server_manager.start_servers(
    >>>     tensor_parallel_size=1,
    >>>     max_num_seqs=1024,
    >>>     gpu_memory_utilization=0.85,
    >>> )
    >>> print(f"Servers started with PIDs/Ports: {server_pids_ports}")
    >>> host_to_http_url = server_manager.host_to_http_url
    >>> # Define vllm_fn() and predict_batch_udf(vllm_fn) and run inference...
    >>> success = server_manager.stop_servers()
    >>> print(f"Server shutdown success: {success}")
    """

    def __init__(self, model_name: str, model_path: str = None):
        super().__init__(model_name, model_path)
        self.vllm_valid_parameters = self._get_valid_vllm_parameters()

    def _get_valid_vllm_parameters(self) -> List[str]:
        """Get valid vLLM parameters on executor."""
        rdd = self.spark.sparkContext.parallelize(list(range(1)), 1)
        return rdd.mapPartitions(lambda _: _get_valid_vllm_parameters_task()).collect()

    def _validate_vllm_kwargs(self, kwargs: Dict[str, Any]):
        """Validate vLLM parameters."""
        for key in kwargs:
            if key not in self.vllm_valid_parameters:
                if key == "host" or key == "port":
                    raise ValueError(
                        f"Invalid vLLM parameter: {key}. Host and port are set by server manager."
                    )
                elif key == "served-model-name":
                    raise ValueError(
                        f"Invalid vLLM parameter: {key}. Served model name is set via model_name."
                    )
                elif key == "model":
                    raise ValueError(
                        f"Invalid vLLM parameter: {key}. Model path is set via model_path."
                    )
                else:
                    raise ValueError(f"Invalid vLLM parameter: {key}")

    def start_servers(
        self,
        wait_retries: int = ServerManager.DEFAULT_WAIT_RETRIES,
        wait_timeout: int = ServerManager.DEFAULT_WAIT_TIMEOUT,
        **kwargs,
    ) -> Dict[str, Tuple[int, List[int]]]:
        """
        Start vLLM OpenAI-compatible servers across the cluster.

        Args:
            wait_retries: Number of retries for waiting for server startup
            wait_timeout: Timeout in seconds for each retry
            **kwargs: Additional arguments to pass to vLLM server command line
                e.g. tensor_parallel_size, max_num_seqs, gpu_memory_utilization, etc.
                See https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#vllm-serve

        Returns:
            Dictionary of hostname -> (server PID, [port])
        """
        self._validate_vllm_kwargs(kwargs)

        return super().start_servers(
            start_server_fn=_start_vllm_server_task,
            wait_retries=wait_retries,
            wait_timeout=wait_timeout,
            **kwargs,
        )
