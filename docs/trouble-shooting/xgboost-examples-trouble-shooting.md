## XGBoost

### 1. NCCL errors

XGBoost supports distributed GPU training which depends on NCCL2 available at [this link](https://developer.nvidia.com/nccl). NCCL auto-detects which network interfaces to use for inter-node communication. If some interfaces are in state up, however are not able to communicate between nodes, NCCL may try to use them anyway and therefore fail during the init functions or **even hang**.

To track NCCL error, User needs to enable NCCL_DEBUG when submitting spark application by 

``` xml
--conf spark.executorEnv.NCCL_DEBUG=INFO
```

Sometimes, Node tries to connect to another node which selects an inappropriate interface, which may cause xgboost task hang. To fix this kind of issue, User needs to specify an appropriate interface for the node by NCCL_SOCKET_IFNAME

``` xml
--conf spark.executorEnv.NCCL_SOCKET_IFNAME=eth0
```