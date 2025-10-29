This document describes how to do [Power Run](../README.md#power-run) on Notebook.  
The approach is binding all the needed parameters to a zip file, then execute the zip file on Notebook.   
First config the files in [configs_in_zip](../configs_in_zip),   
then compress the [nds folder](../../nds) into a zip,  
Finally run the `nds_power.py` in the zip on Notebook.
# config parameter files
## parameter file
e.g.:
```
s3://ndsv2-data/parquet_sf1000 time-123.csv --output_prefix s3://chongg/test/tmp --json_summary_folder 500 --keep_sc
```
`s3://ndsv2-data/parquet_sf1000` is the value of `input_prefix` parameter  
`time-123.csv` is the value of `time_log` parameter  
`s3://chongg/test/tmp` is the value of `output_prefix` parameter  
`500` is the value of `json_summary_folder` parameter  
Note: please do not specify `property_file`, refer to the next section to config Spark property file.  
For more details, refer to the parameters in the [Power Run](../README.md#power-run)  
## spark.properties file
Specify the Spark properties.
e.g.:
```
spark.executor.memoryOverhead=512M
```
## query_0.sql
This file is the stream file.  
Put all the queries into this file.  
For how to generate stream file, refer to [README](../README.md).
# create NDS zip file and put it into S3
Do the following commands to zip all the nds folder into a zip file.
```
cd spark-rapids-benchmarks
zip -r nds.zip nds
aws s3 cp nds.zip s3://path/to/this/zip
```
# run nds_power on notebook
```
spark.sparkContext.addPyFile("s3://path/to/this/zip")
from nds import nds_power
nds_power.run_query_stream_for_zip()
```
# How to run another Power RUN with different parameters
Updating the zip file in S3 does not take effective because Spark have caches for the zip file.    
You can put another zip file in to S3.    
Do the following:  
```
cd spark-rapids-benchmarks
cp -r nds <another_nds_name>
# update the parameter files
zip -r <another_nds_name>.zip <another_nds_name>
aws s3 cp <another_nds_name>.zip s3://<another nds zip path>
```
On Notebook run:
```
spark.sparkContext.addPyFile("s3://<another nds zip path>")
from <another_nds_name> import nds_power
nds_power.run_query_stream_for_zip()
```