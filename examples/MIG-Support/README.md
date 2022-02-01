# Multi-Instance GPU (MIG) support in Apache Hadoop YARN

There are multiple solutions for MIG scheduling on YARN that you can choose based on your environment and
deployment requirements:

- [YARN 3.3.0+ MIG GPU Plugin](/examples/MIG-Support/device-plugins/gpu-mig) for adding a Java-based plugin for MIG
on top of the Pluggable Device Framework
- [YARN 3.1.2 until YARN 3.3.0 MIG GPU Support](/examples/MIG-Support/resource-types/gpu-mig) for
patching and rebuilding YARN code base to support MIG devices.
- [YARN 3.1.2+ MIG GPU Support without modifying YARN / Device Plugin Code](/examples/MIG-Support/yarn-unpatched)
relying on installing nvidia CLI wrappers written in `bash`, but unlike the solutions above without
any Java code changes.

## Limitations and Caveats

Note that are some common caveats for the solutions above.

### Single MIG GPU per Container

Please see the [MIG Application Considerations](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#app-considerations)
and [CUDA Device Enumeration](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html#cuda-visible-devices).

It is important to note that CUDA 11 only supports enumeration of a single MIG instance.
It is recommended that you configure YARN to only allow a single GPU be requested. See
the YARN config `yarn.resource-types.nvidia/miggpu.maximum-allocation` for the [Pluggable Device Framework]
(/examples/MIG-Support/device-plugins/gpu-mig) solution and
`yarn.resource-types.yarn.io/gpu.maximum-allocation` for the remainder of MIG Support options above, respectively.

### Metrics
Some metrics are not and cannot be broken down by MIG device. For example, `utilization` is the
aggregate utilization of the parent GPU, and there is no attribution of `temperature` to a
particular MIG device.

### GPU index / address as reported by Apache Spark in logs and UI

With YARN isolation using NVIDIA Container Runtime ensuring a single visible device
per Docker container running a Spark Executor, each Executor will see a disjoint list comprising
a single device.
Therefore, the user will end up observing index 0 being used by all executors. However, they refer
to different GPU/MIG instances. You can verify this by running something like the following on a
YARN worker node host OS:

```bash
for cid in $(sudo docker ps -q); do sudo docker exec $cid bash -c "printenv | grep VISIBLE; nvidia-smi -L"; done
NVIDIA_VISIBLE_DEVICES=3
GPU 0: NVIDIA A30 (UUID: GPU-05aa99be-b706-0dc1-ab62-dd12f2227b7d)
  MIG 1g.6gb      Device  0: (UUID: MIG-70dc024a-e8d7-587c-81dd-57ad493b1d91)
NVIDIA_VISIBLE_DEVICES=1
GPU 0: NVIDIA A30 (UUID: GPU-05aa99be-b706-0dc1-ab62-dd12f2227b7d)
  MIG 1c.2g.12gb  Device  0: (UUID: MIG-54cc2421-6f2d-59e9-b074-20707aadd71e)
NVIDIA_VISIBLE_DEVICES=2
GPU 0: NVIDIA A30 (UUID: GPU-05aa99be-b706-0dc1-ab62-dd12f2227b7d)
  MIG 1g.6gb      Device  0: (UUID: MIG-7e5552bf-d328-57a8-b091-0720d4530ffb)
NVIDIA_VISIBLE_DEVICES=0
GPU 0: NVIDIA A30 (UUID: GPU-05aa99be-b706-0dc1-ab62-dd12f2227b7d)
  MIG 1c.2g.12gb  Device  0: (UUID: MIG-e6af58f0-9af8-594f-825e-74d23e1a68c1)
```




