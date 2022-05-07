# telco-churn-augmentation

This demo is derived from [data-science-blueprints](https://github.com/NVIDIA/data-science-blueprints) repository.
The repository shows a realistic ETL workflow based on synthetic normalized data.  It consists of two pieces:

1.  _an augmentation notebook_, which synthesizes normalized (long-form) data from a wide-form input file,
    optionally augmenting it by duplicating records, and
2. _an ETL notebook_, which performs joins and aggregations in order to generate wide-form data from the synthetic long-form data.

From a performance evaluation perspective, the latter is the interesting workload; the former is just a data generator for the latter.

## Running as notebooks

The notebooks ([`augment.ipynb`](./notebooks/python/augment.ipynb) and [`etl.ipynb`](./notebooks/python/etl.ipynb)) are the best
resource to understand the code and can be run interactively or with Papermill.  
The published Papermill parameters are near the top of each notebook.

