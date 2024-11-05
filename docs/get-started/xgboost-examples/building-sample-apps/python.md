# Build XGBoost Python Examples

## Build

Follow these steps to package the Python zip file:

``` bash
git clone https://github.com/NVIDIA/spark-rapids-examples.git
cd spark-rapids-examples/scripts/building
sh python_build.sh
```


## Files Required by PySpark

Two files are required by PySpark:

+ *samples.zip*
  
  the package including all example code. 
  Executing the above build commands generates the samples.zip file in 'spark-rapids-examples/examples/XGBoost-Examples' folder

+ *main.py*
  
  entrypoint for PySpark, you can find it in 'spark-rapids-examples/examples/XGBoost-Examples' folder
