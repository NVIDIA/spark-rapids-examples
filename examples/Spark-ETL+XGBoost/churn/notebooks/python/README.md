# telco-churn-augmentation

This demo shows a realistic ETL workflow based on synthetic normalized data.  It consists of two pieces:

1.  _an augmentation script_, which synthesizes normalized (long-form) data from a wide-form input file, 
    optionally augmenting it by duplicating records, and
2. _an ETL script_, which performs joins and aggregations in order to generate
   wide-form data from the synthetic long-form data.

From a performance evaluation perspective, the latter is the interesting workload; 
the former is just a data generator for the latter.
