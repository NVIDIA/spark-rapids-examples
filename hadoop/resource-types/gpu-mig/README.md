# NVIDIA Support for GPU for YARN with MIG support for Hadoop 3.1.2 until Hadoop 3.3.0

This patch adds support for GPUs with [MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) on YARN for Hadoop versions 3.1.2 until 3.3.0 (not including 3.3.0). Use the [GPU Plugin for YARN with MIG support](../device-plugins/gpu-mig/README.md) for Hadoop 3.3.0 and newer versions.
The built-in YARN GPU plugin does not support MIG enabled GPUs so this patch is needed to properly support.
This patch also works with GPUs without MIG or GPUs with MIG disabled but the limitation section still applies. It supports heterogenous
environments where there may be some MIG enabled GPUs and some without MIG.
This requires patching YARN and rebuilding it.

## Compatibility

It works with Apache YARN 3.1.2 until 3.3.0 versions that support GPU scheduling. MIG support requires YARN to be configured with cgroups and NVIDIA Docker runtime v2.

## Limitations

Please see the [MIG Application Considerations](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#app-considerations)
and [CUDA Device Enumeration](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#cuda-visible-devices).

It is important to note that CUDA 11 only supports enumeration of a single MIG instance. This means that with this patch 
and MIG support enabled, it only supports 1 GPU per container and will throw an exception by default if you request more.
It is recommended that you configure YARN to only allow a single GPU be requested. See the yarn config:
```
 yarn.resource-types.yarn.io/gpu.maximum-allocation
```
See [YARN Resource Configuration](https://hadoop.apache.org/docs/r3.1.2/hadoop-yarn/hadoop-yarn-site/ResourceModel.html) for more details.
If you do not configure the maximum allocation and someone requests multiple GPUs, the default behavior is to throw an exception.
See the [Configuration](#configuration) section for options if it throws an exception.

## Building
Apply the patch to your YARN version and build it like you would normally for your deployment.

For example:
```
patch -p1 < hadoop312MIG.patch
mvn clean package -Pdist -Dtar -DskipTests
```

Run unit tests:
```
mvn test -Pdist -Dtar -Dtest=TestGpuDiscoverer
mvn test -Pdist -Dtar -Dtest=TestNvidiaDockerV2CommandPlugin
```

## Installation

This assumes YARN was already installed and configured with GPU scheduling and cgroups enabled and Nvidia Docker runtime v2.
See [Using GPU on YARN](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/UsingGpus.html) if you need more information. 

Enable and configure your [GPUs with MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html) on all of the nodes it applies to.

Install the new YARN version built with the patch on your YARN Cluster.

Enable the MIG GPU support in the Hadoop configuration files:

```
<property>
  <name>yarn.nodemanager.resource-plugins.gpu.use-mig-enabled</name>
  <value>true</value>
</property>

```

Restart YARN if needed to pick up any configuration changes.

## Configuration

The default behavior of the GPU resource plugin on YARN is to use `auto` discovery mode of GPUs on each nodemanager.
It also allows you to manually allow certain gpu devices. This configuration was extended to support MIG devices.
`yarn.nodemanager.resource-plugins.gpu.allowed-gpu-devices` configuraton can be used to manually specify devices.
GPU device is identified by their minor device number, index, and optionally MIG device index. A common approach to get
minor device number of GPUs is using nvidia-smi -q and search Minor Number output and optionally MIG device indices.
The format is index:minor_number[:mig_index][,index:minor_number...]. An example of manual specification is
0:0,1:1:0,1:1:1,2:2" to allow YARN NodeManager to manage GPU devices with indices 0/1/2 and minor number 0/1/2
where GPU indices 1 has 2 MIG enabled devices with indices 0/1.
```
<property>
  <name>yarn.nodemanager.resource-plugins.gpu.allowed-gpu-devices</name>
  <value>0:0,1:1:0,1:1:1,2:2</value>
</property>
```

To change the behavior of throwing when the user allocates multiple GPUs can be controlled by setting an environment variable
when the Spark application is launched. Setting it to `true` means to throw if a user requests multiple GPUs (this is the default), `false`
means it won't throw and if the container is allocated with multiple MIG devices from the same GPU, it is up to the
application to know how to use them.

Environment variable for Spark application:
```
--conf spark.executorEnv.NVIDIA_MIG_PLUGIN_THROW_ON_MULTIPLE_GPUS=false
```

## Testing
Run a Spark application using the [Rapids Accelerator for Apache Spark](https://nvidia.github.io/spark-rapids/) and request GPUs
from YARN and verify they use the MIG enabled GPUs.
