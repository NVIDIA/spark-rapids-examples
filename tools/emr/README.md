# EMR Tools Demo Notebooks

The RAPIDS Accelerator for Apache Spark includes two key tools for understanding the benefits of
GPU acceleration as well as analyzing GPU Spark jobs.  For customers on EMR, the demo
notebooks offer a simple interface for running the tools given a set of Spark event logs from
CPU (qualification) or GPU (profiling) application runs.

To use a demo notebook, you can import the notebook in the EMR Workspace.

Once the demo notebook is imported, you can enter in the log path location in the cell below the `User Input` in the 
notebook.  After that, click on the `fast-forward` icon which says *Restart the kernel, then re-run the whole notebook* to execute the tools for the specific logs in the log path.

## Limitations
1. Currently, local and S3 event log paths are supported.
1. Eventlog path must follow the formats `/local/path/to/eventlog` for local logs or `s3://my-bucket/path/to/eventlog` for logs stored in S3.
1. The specified path can also be a directory. In such cases, the tool will recursively search for event logs within the directory.
   - For example: `/path/to/clusterlogs`
1. To specify multiple event logs, separate the paths with commas.
   - For example: `s3://my-bucket/path/to/eventlog1,s3://my-bucket/path/to/eventlog2`

**Latest Tools Version Supported** 24.08.2