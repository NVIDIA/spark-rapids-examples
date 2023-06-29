
/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.examples.utility

import com.google.common.base.CaseFormat

import scala.collection.mutable
import scala.util.Try

import ml.dmlc.xgboost4j.scala.spark.TrackerConf

private case class XGBoostArg(
  required: Boolean = false,
  parse: String => Any = value => value,
  message: String = "")

object XGBoostArgs {
  private val modes = Seq("all", "train", "transform")
  private val formats = Seq("csv", "parquet", "orc")
  private val stringToBool = Map(
    "true"  -> true,
    "false" -> false,
    "1" -> true,
    "0" -> false
  )
  private val booleanMessage = "Expect 'true' or '1' for true, 'false' or '0' for false."

  private def parseDataRatios(value: String): (Int, Int) = {
    val ratios = value.split(":").filter(_.nonEmpty).map(_.toInt)
    require(ratios.length == 2 && ratios(0) + ratios(1) <= 100)
    (ratios(0), ratios(1))
  }

  private val supportedArgs = Map(
    "mode"   -> XGBoostArg(
      parse = value => { require(modes.contains(value)); value },
      message = s"Expect one of [${modes.mkString(", ")}]"),
    "format" -> XGBoostArg(true,
      parse = value => { require(formats.contains(value)); value },
      message = s"Expect one of [${formats.mkString(", ")}]"),
    "dataPath"  -> XGBoostArg(true),
    "dataRatios" -> XGBoostArg(
      parse = parseDataRatios,
      message = "Expect as <train>:<transform>, both train and transform require Int, and total value <= 100"),
    "modelPath" -> XGBoostArg(),
    "numRows"   -> XGBoostArg(parse = _.toInt, message = "Require an Int."),
    "numFold" -> XGBoostArg(parse = _.toInt, message = "Require an Int."),
    "showFeatures" -> XGBoostArg(parse = stringToBool, message = booleanMessage),
    "overwrite" -> XGBoostArg(parse = stringToBool, message = booleanMessage),
    "hasHeader" -> XGBoostArg(parse = stringToBool, message = booleanMessage),
    "saveDict"  -> XGBoostArg(parse = stringToBool, message = booleanMessage),
  )

  private def help: Unit = {
    println("\n\nSupported arguments:")
    println("    -dataPath=path: String, Required\n" +
      "        The path of data file(s). Use multiple '-dataPath=path#' to specify multiple paths. Such as" +
      " '-dataPath=path1 -dataPath=path2'.\n")
    println("    -format=<csv/parquet/orc>: String, Required\n" +
      "        The format of the data, now only supports 'csv', 'parquet' and 'orc'.\n")
    println("    -mode=<all/train/transform>: String\n" +
      "        To control the behavior of apps. Default is 'all'. \n" +
      "        * all: Do training and transformation.\n" +
      "        * train: Do training only, will save model to 'modelPath' if specified.\n" +
      "        * transform: Transformation only, 'modelPath' is required to provide the model.\n")
    println("    -modelPath=path: String\n" +
      "        Specify where to save model after training, or where to load model for transforming only. \n")
    println("    -overwrite=value: Boolean\n" +
      "        Whether to overwrite the current model data under 'modelPath'. Default is false\n")
    println("    -dataRatios=train<Int>:transform<Int>\n" +
      "        The ratios of data used for train and transform, then the ratio for evaluation is (100-train-test)." +
      " default is 80:20, no evaluation\n")
    println("    -hasHeader=value: Boolean\n" +
      "        Whether the csv file has header. Default is true.\n")
    println("    -numRows=value: Int\n" +
      "        Number of the rows to show after transformation. Default is 5.\n")
    println("    -numFold=value: Int\n" +
      "        Number of the folders to be used in Cross Validation. Default is 3.\n")
    println("    -showFeatures=value: Boolean\n" +
      "        Whether to include the features columns when showing results of transformation. Default is true.\n")
    println("    -saveDict=value: Boolean\n" +
      "        Whether to save the dictionary table for Mortgage ETL. It is saved under '<out>/.dict'. Default is true.\n")
    println("    -rabitTrackerHost=value: String\n" +
      "        Specify rabit tracker host IP address. In some environments XGBoost might fail to resolve\n" +
               "the IP address of the rabit tracker, a symptom is user receiving ``OSError: [Errno 99]\n" +
               "Cannot assign requested address`` error during training.  A quick workaround is to\n" +
               "specify the address explicitly.\n")
    println("For XGBoost arguments:")
    println("    Now we pass all XGBoost parameters transparently to XGBoost, no longer to verify them.")
    println("    Both of the formats are supported, such as 'numWorkers'. You can pass as either one below:")
    println("    -numWorkers=10  or  -num_workers=10 ")
    println()
  }

  def apply(args: Array[String]) = parse(args)

  def parse(args: Array[String]): XGBoostArgs = {
    val appArgsMap = mutable.HashMap.empty[String, Any]
    val xgbArgsMap = mutable.HashMap.empty[String, String]
    try {
      args.filter(_.nonEmpty).foreach {
        argString =>
          require(argString.startsWith("-") && argString.contains('='),
            s"Invalid argument: $argString, expect '-name=value'")

          val parts = argString.stripPrefix("-").split('=').filter(_.nonEmpty)
          require(parts.length == 2, s"Invalid argument: $argString, expect '-name=value'")

          val (key, value) = (parts(0), parts(1))
          if (supportedArgs.contains(key)) {
            // App arguments
            val parseTry = Try(supportedArgs(key).parse(value))
            require(parseTry.isSuccess,
              s"Invalid value to '$key'. ${supportedArgs(key).message}")
            if (key == "dataPath") {
              val paths = appArgsMap.getOrElse(key, Seq.empty).asInstanceOf[Seq[String]] :+ parseTry.get
              appArgsMap += key -> paths
            } else {
              appArgsMap += key -> parseTry.get
            }
          } else {
            // Supposed to be XGBooost parameters
            xgbArgsMap += key -> value
          }
      }
      supportedArgs.filter(_._2.required).foreach {
        case (name, _) => require(appArgsMap.contains(name), s"Missing argument: $name.")
      }
      new XGBoostArgs(appArgsMap.toMap, xgbArgsMap.toMap)
    } catch {
      case e: Exception =>
        help
        throw e
    }
  }
}

class XGBoostArgs private[utility] (
    val appArgsMap: Map[String, Any],
    val xgbArgsMap: Map[String, String]) {

  def format: String = appArgsMap("format").asInstanceOf[String]

  def modelPath: Option[String] = appArgsMap.get("modelPath").asInstanceOf[Option[String]]

  // mode is optional with default value 'all'
  private def mode: String = appArgsMap.getOrElse("mode", "all").asInstanceOf[String]

  private[utility] def verifyArgsRelation: Unit = {
    if (mode == "train" && modelPath.isEmpty) {
      println("==> You may want to specify the 'modelPath' to save the model when 'train only' mode.")
    }
    if (mode == "transform") {
      require(modelPath.nonEmpty, "'modelPath' is required for mode: transform")
    }
  }
  verifyArgsRelation

  def isToTrain: Boolean = mode != "transform"
  def isToTransform: Boolean = mode != "train"

  def dataPaths: Seq[String] = appArgsMap("dataPath").asInstanceOf[Seq[String]]

  def dataRatios: (Int, Int, Int) = {
    val ratios = appArgsMap.get("dataRatios").asInstanceOf[Option[(Int, Int)]].getOrElse((80, 20))
    (ratios._1, ratios._2, 100 - ratios._1 - ratios._2)
  }

  def isShowFeatures: Boolean = appArgsMap.get("showFeatures").forall(_.asInstanceOf[Boolean])

  def isOverwrite: Boolean = appArgsMap.get("overwrite").exists(_.asInstanceOf[Boolean])

  def hasHeader: Boolean = appArgsMap.get("hasHeader").forall(_.asInstanceOf[Boolean])

  def saveDict: Boolean = appArgsMap.get("saveDict").forall(_.asInstanceOf[Boolean])

  def numRows: Int = appArgsMap.get("numRows").asInstanceOf[Option[Int]].getOrElse(5)

  def numFold: Int = appArgsMap.get("numFold").asInstanceOf[Option[Int]].getOrElse(3)

  def xgboostParams(otherParams: Map[String, Any] = Map.empty): Map[String, Any] = {
    val params = otherParams ++ xgbArgsMap.map{
        case (name, value) if !name.contains('_') =>
          (CaseFormat.LOWER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, name), value)
        case (name, value) => (name, value)
    }

    val hostIp = params.getOrElse("rabit_tracker_host", "").toString
    if (!hostIp.isEmpty) {
      params ++ Map("tracker_conf" -> TrackerConf(0l, hostIp))
    } else params
  }

  /**
   *  getDataPaths check and get train/eval/transform paths
   * @return Array(train_paths, eval_paths, transform_paths)
   */
  def getDataPaths: Array[Seq[String]] = {
    val paths = dataPaths
    val prefixes = Array("train::", "eval::", "trans::")
    val validPaths = paths.filter(_.nonEmpty).map(_.trim)

    // get train data paths
    val trainPaths = validPaths.filter(_.startsWith(prefixes.head))
    if (isToTrain) {
      require(trainPaths.nonEmpty, s"requires at least one path for train file." +
        s" Please specify it by '-dataPath=train::your_train_data_path'")
    }

    // get eval path
    val evalPaths = validPaths.filter(_.startsWith(prefixes(1)))

    // get and check train data paths
    val transformPaths = validPaths.filter(_.startsWith(prefixes(2)))
    if (isToTransform) {
      require(transformPaths.nonEmpty, s"requires at least one path for transform file." +
        s" Please specify it by '-dataPath=trans::your_transform_data_path'")
    }

    // check data paths not specified type
    val unknownPaths = validPaths.filterNot(p => prefixes.exists(p.contains(_)))
    require(unknownPaths.isEmpty, s"Unknown type for data path: ${unknownPaths.head}, requires to specify" +
      " the type for each data path by adding the prefix 'train::' or 'eval::' or 'trans::'.")

    Array(trainPaths.map(_.stripPrefix(prefixes.head)),
      evalPaths.map(_.stripPrefix(prefixes(1))),
      transformPaths.map(_.stripPrefix(prefixes(2))))
  }
}
