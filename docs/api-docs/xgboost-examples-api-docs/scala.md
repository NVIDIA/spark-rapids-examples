# Scala API for XGBoost-Spark3.0

This doc focuses on GPU related Scala API interfaces, and fortunately only one new API is introduced to support training on GPU.

XGBoost-Spark3.0 provides four classes as below to support ML things on spark:

- [XGBoostClassifier](#xgboostclassifier)
- [XGBoostClassificationModel](#xgboostclassificationmodel)
- [XGBoostRegressor](#xgboostregressor)
- [XGBoostRegressionModel](#xgboostregressionmodel)

### XGBoostClassifier

The full name is ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier. It extends [ProbabilisticClassifier](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.classification.ProbabilisticClassifier)[[Vector](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.linalg.Vector), [XGBoostClassifier](#xgboostclassifier), [XGBoostClassificationModel](#xgboostclassificationmodel)].

#####  Constructors

+ XGBoostClassifier(xgboostParams: Map[String, Any])
    + all [standard xgboost parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) are supported
    + eval_sets: Map[String,DataFrame] is used to set the named evaluation dataset(s) for training.

##### Methods

*Note: Only GPU related methods are listed below.*

+ setFeaturesCols(value: Seq[String]): [XGBoostClassifier](#xgboostclassifier). This method sets the feature columns for training.
    + value: a sequence of feature column name
    + returns the classifier itself

### XGBoostClassificationModel

The full name is ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel. It extends [ProbabilisticClassificationModel](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.ProbabilisticClassificationModel)[[Vector](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.linalg.Vector), [XGBoostClassificationModel](#xgboostclassificationmodel)].

##### Methods

*No GPU specific methods, use it as a normal spark model.*

### XGBoostRegressor

The full name is ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor. It extends [Predictor](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.Predictor)[[Vector](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.linalg.Vector), [XGBoostRegressor](#xgboostregressor), [XGBoostRegressionModel](#xgboostregressionmodel)].

#####  Constructors

+ XGBoostRegressor(xgboostParams: Map[String, Any])
    + all [standard xgboost parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) are supported
    + eval_sets: Map[String,DataFrame] is used to set the named evaluation dataset(s) for training.

##### Methods

*Note: Only GPU related methods are listed below.*

+ setFeaturesCols(value: Seq[String]): [XGBoostRegressor](#xgboostregressor). This method sets the feature columns for training.
    + value: a sequence of feature column names to set
    + returns the regressor itself

### XGBoostRegressionModel

The full name is ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel. It extends [PredictionModel](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.PredictionModel)[[Vector](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.linalg.Vector), [XGBoostRegressionModel](#xgboostregressionmodel)].

##### Methods

*No GPU specific methods, use it as a normal spark model.*


