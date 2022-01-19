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

Download the contents of [scripts](./scripts/) to every YARN NodeManager (worker) machine
to some location, for example: `/usr/local/yarn-mig-scripts`. Note both wrappers leave
the original outputs untouched if the environment variable `MIG_AS_GPU_ENABLED` is not 1.

### YARN Configuration

Add `export MIG_AS_GPU_ENABLED=1` to `$YARN_CONF_DIR/yarn-env.sh`.
Customize `REAL_NVIDIA_SMI_PATH` if not at the default location `/usr/bin/nvidia-smi`.

Modify the following config `$YARN_CONF_DIR/yarn-site.xml`:
```xml
<property>
  <name>yarn.nodemanager.resource-plugins.gpu.path-to-discovery-executables</name>
  <value>/usr/local/yarn-mig-scripts</value>
</property>
```

If you disabled the default automatic GPU discovery set
`yarn.nodemanager.resource-plugins.gpu.allowed-gpu-devices` to the 0-based index of the GPU element
in the XML output of the manual invocation:

```bash
MIG_AS_GPU_ENABLED=1 /usr/local/yarn-mig-scripts -q -x
```

### NVIDIA Docker Runtime Configuration

Modify section `[nvidia-container-cli]` in `/etc/nvidia-container-runtime/config.toml`:
```toml
path = "/usr/local/yarn-mig-scripts/nvidia-container-cli-wrapper.sh"
environment = [ "MIG_AS_GPU_ENABLED=1",  "REAL_NVIDIA_SMI_PATH=/if/non-default/path/nvidia-smi" ]
```





