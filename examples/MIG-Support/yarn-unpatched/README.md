# MIG Support for Spark on YARN using unmodified versions of Apache Hadoop 3.1.2+

This document describes a solution for utilizing MIG with YARN when upgrading to a recent 3.3+
version or patching older versions of Apache Hadoop is not feasible. Please refer to the corresponding
alternatives for more information:
- [Device Plugins README](../device-plugins/gpu-mig/README.md)
- [YARN patch README](../resource-types/gpu-mig/README.md)

## Introduction

We provide a set of scripts that wrap the original `nvidia-smi` from the NVIDIA GPU Driver and `nvidia-container-cli`
included in [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

`nvidia-smi-wrapper.sh` is a wrapper script that parses the XML output of `nvidia-smi -q -x` used by YARN
to discover GPUs. It replaces MIG-enabled GPUs with the list of `<gpu>` elements corresponding to every
`<mig_device>` element of the GPU with additional annotation to construct the MIG identifier for
`nvidia-container-cli`. This reverse mapping is performed by  modified `nvidia` Docker runtime using
`nvidia-container-cli-wrapper.sh`.

## Installation

These instructions assume NVIDIA Container Toolkit (nvidia-docker2) and YARN is already installed
and configured with [CGroups enabled](https://hadoop.apache.org/docs/r3.1.2/hadoop-yarn/hadoop-yarn-site/UsingGpus.html).

Enable and configure your [GPUs with MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html) on all of the nodes
it applies to.

Download the contents of [scripts](./scripts/) to every YARN NodeManager (worker) machine
to some location, for example: `/usr/local/yarn-mig-scripts`. Make sure that the scripts
are executable by the docker daemon user (i.e., `root`), and YARN NM service user (typically `yarn`). Note that the scripts
leave the original outputs untouched if the environment variable `MIG_AS_GPU_ENABLED` is not 1.

### YARN Configuration

Add `export MIG_AS_GPU_ENABLED=1` to `$YARN_CONF_DIR/yarn-env.sh`.
Customize `REAL_NVIDIA_SMI_PATH` if not at the default location `/usr/bin/nvidia-smi`.

Modify the following config `$YARN_CONF_DIR/yarn-site.xml`:
```xml
<property>
  <name>yarn.nodemanager.resource-plugins.gpu.path-to-discovery-executables</name>
  <value>/usr/local/yarn-mig-scripts/nvidia-smi-wrapper.sh</value>
</property>
```

By default, `yarn.nodemanager.resource-plugins.gpu.allowed-gpu-devices` is set to `auto` and
and `/usr/local/yarn-mig-scripts/nvidia-smi-wrapper.sh` will be called by YARN to discover GPUs.

If you disable the default automatic GPU discovery, you can manually
specify the list of MIG instances to use by setting
`yarn.nodemanager.resource-plugins.gpu.allowed-gpu-devices` to the list of
0-based indices corresponding to the desired `<gpu>` elements in the output of

```bash
MIG_AS_GPU_ENABLED=1 /usr/local/yarn-mig-scripts/nvidia-smi-wrapper.sh -q -x
```

In other words, if you want to allow MIG 1:2 and 2:0 and they are listed as 3rd and 5th `<gpu>`
elements the value for `yarn.nodemanager.resource-plugins.gpu.allowed-gpu-devices` should be
"2,4".

### NVIDIA Docker Runtime Configuration

Modify section `[nvidia-container-cli]` in `/etc/nvidia-container-runtime/config.toml`:
```toml
path = "/usr/local/yarn-mig-scripts/nvidia-container-cli-wrapper.sh"
environment = [ "MIG_AS_GPU_ENABLED=1",  "REAL_NVIDIA_SMI_PATH=/if/non-default/path/nvidia-smi" ]
```

## Limitations and Caveats
Some metrics are not and cannot be broken down by MIG device. For example, `utilization` is the
aggregate utilization of the parent GPU, and there is no attribution of `temperature` to a
particular MIG device.