# Build XGBoost Python Examples

## Build

Follow these steps to package the Python zip file:

``` bash
git clone https://gitlab-master.nvidia.com/nvspark/spark-examples2.git
cd spark-examples2/scripts/building
sh python_build.sh
```


## Files Required by PySpark

Two files are required by PySpark:

+ *samples.zip*
  
  the package including all example code

+ *main.py*
  
  entrypoint for PySpark, you can find it in 'spark-xgboost-examples/examples' folder
