# Python API for XGBoost-Spark

This doc focuses on GPU related Python API interfaces. Four new classes are introduced to support ML things on spark.

- [XGBoostClassifier](#xgboostclassifier)
- [XGBoostClassificationModel](#xgboostclassificationmodel)
- [XGBoostRegressor](#xgboostregressor)
- [XGBoostRegressionModel](#xgboostregressionmodel)


### XGBoostClassifier

The full name is ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier. It is a wrapper around [Scala XGBoostClassifier](scala.md#xgboostclassifier).

#####  Constructors

+ XGBoostClassifier(\*\*params)
    + all [standard xgboost parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) are supported, but please note a few differences:
        + only camelCase is supported when specifying parameter names, e.g., *maxDepth*
        + parameter *lambda* is renamed to *lambda_*, because *lambda* is a keyword in Python

##### Methods

*Note: Only GPU related methods are listed below.*

+ setFeaturesCols(features_cols). This method sets the feature columns for training.
    + features_cols: a list of feature column names in String format to set
    + returns the classifier itself

### XGBoostClassificationModel

The full name is ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel. It is a wrapper around [Scala XGBoostClassificationModel](scala.md#xgboostclassificationmodel).

##### Methods

*No GPU specific methods, use it as a normal spark model.*

### XGBoostRegressor

The full name is ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor. It is a wrapper around [Scala XGBoostRegressor](scala.md#xgboostregressor).

#####  Constructors

+ XGBoostRegressor(\*\*params)
    + all [standard xgboost parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) are supported, but please note a few differences:
        + only camelCase is supported when specifying parameter names, e.g., *maxDepth*
        + parameter *lambda* is renamed to *lambda_*, because *lambda* is a keyword in Python

##### Methods

*Note: Only GPU related methods are listed below.*

+ setFeaturesCols(features_cols). This method sets the feature columns for training.
    + features_cols: a list of feature column names in String format to set
    + returns the regressor itself

### XGBoostRegressionModel

The full name is ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel. It is a wrapper around [Scala XGBoostRegressionModel](scala.md#xgboostregressionmodel).

##### Methods

*No GPU specific methods, use it as a normal spark model.*

