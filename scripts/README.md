### Encoding Tool
This tool is to convert the values from categorical type to numerical type in certain columns. Currently we supoort `mean encoding` and `one-hot encoding`.

### Main Procedure
1. User should firstly use our tool to profile the raw data source to get a "dictinary"(We call this dictionary `model`) that maps categorical values to certain numerical values. We call this method `train`. Each column will have its own `model`
2. User will use the `model` they got from step 1 to replace those categorical values with numerical values. 

### Usage
1. `cd encoding/python`
2. `zip -r sample.zip com` to get a python encoding tool library
3. submit the encoding job to your Spark host

You can find full use cases in `encoding-sample/run.sh`

### Application Parameters
 - mainClass: 
   
   - `com.nvidia.spark.encoding.criteo.one_hot_cpu_main`: one-hot encoding
   - `com.nvidia.spark.encoding.criteo.target_cpu_main`: target(mean) encoding
 - mode: 
   - `train`: use raw data to get encoding model
   - `transform`: use encoding moddel to convert raw data
 - format:
   - `csv`: only csv is supported
 - columns: 
   - the target columns user wants to convert, e.g. `_34,_35` means user wants to get dictionary for both `_34` and `_35` columns
 - modelPaths: 
   - for `train` mode, it points to the path where user wants to save the encoding model
   - for `transform` mode, it points to the model that the encoding conversion needs.
   - it is 1-1 mapped to `columns`. If user wants to encode 2 columns, he must provide 2 `modelPaths`. e.g. `model_34,model_35`
 - inputPaths: 
   - raw data user wants to get encoding model from, or to convert
 - outputPaths: 
   - only used in `transform` mode.
 - overwrite:
   - whether overwrite the exsiting model or output data
 - numRows:
   - optinal. show some rows in command line when encoding is finished. 
 - labelColumn:
   - required in `target encoding`. Set the label column of raw data.

### Optimization
1. Due to default behaviors from some Spark methods, Some value may contain useless precison which causes the large size of `model`.e.g. 0.000000 and 1.000000 are identical to 0 and 1 in value perspective, but the csv model file that contains those values costs more disk space. We provide `truncate-model.py` in `encoding-sample` to remove the extra useless precisions.
2. We provide a repartition kit `repartition.py` to reparitition your output data.

The usage can also be found in `encoding-sample/run.sh`