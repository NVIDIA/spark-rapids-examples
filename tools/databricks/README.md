# Databricks Tools Demo Notebooks

The RAPIDS Accelerator for Apache Spark includes two key tools for understanding the benefits of
GPU acceleration as well as analyzing GPU Spark jobs.  For customers on Databricks, the demo
notebooks offer a simple interface for running the tools given a set of Spark event logs from
CPU (qualification) or GPU (profiling) application runs.

To use a demo notebook, you can import the notebook in the Databricks Notebook UI via File->Import Notebook.

Once the demo notebook is imported, you can select run to activate the notebook to an available compute
cluster.  Once the notebook is activated, you can enter in the log path location in the text widget at the
top of the notebook.  After that, select *Run all* to execute the tools for the specific logs in the log path.

## Limitations
1. Currently only local or DBFS eventlog paths are supported.
2. DBFS path must use the File API Format. 
3. Example: `/dbfs/<path-to-event-log>`.

**Latest Tools Version Supported** 24.04.0