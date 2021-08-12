Preparing Datasets
==================

## Mortgage Example
1. Setup [Apache Toree Jupyter notebook](/docs/get-started/xgboost-examples/notebook/toree.md).
2. Download raw data from: https://rapidsai.github.io/demos/datasets/mortgage-data
3. Run [Mortgage ETL job](/examples/mortgage/notebooks/python/MortgageETL.ipynb).

## Taxi Example
1. Setup [Apache Toree Jupyter notebook](/docs/get-started/xgboost-examples/notebook/toree.md).
2. Install `cudatoolkit` and `numba` (`conda` example provided, but you can also use `pip`):
```
conda install numba
conda install cudatoolkit
```
3. Download raw data:
```
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_20{09..16}-{01..12}.csv
```
4. Run [Taxi ETL job](/examples/taxi/notebooks/python/Taxi_ETL.ipynb).
