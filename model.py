# =============================================================================
# IMPORTS
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd
import numpy as np
import os
import sys


# =============================================================================
# 1. SPARK SESSION
# =============================================================================

def create_spark_session(app_name="CabRidePricePrediction"):
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(f"✅ Spark Session Started: {spark.version}")
    return spark


# =============================================================================
# 2. DATA LOADING
# =============================================================================

def load_data(spark, file_path):
    df = spark.read.csv(
        file_path,
        header=True,
        inferSchema=True
    )
    print(f"✅ Loaded {df.count():,} rows | {len(df.columns)} columns")
    return df


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def extract_datetime_features(df, timestamp_col="time_stamp"):
    if timestamp_col not in df.columns:
        return df

    df = df.withColumn(
        "timestamp_converted",
        (F.col(timestamp_col) / 1000).cast("timestamp")
    )

    df = df.withColumn("hour", F.hour("timestamp_converted")) \
           .withColumn("day_of_week", F.dayofweek("timestamp_converted")) \
           .withColumn("month", F.month("timestamp_converted"))

    df = df.withColumn(
        "time_of_day",
        F.when(F.col("hour").between(6, 11), "morning")
         .when(F.col("hour").between(12, 16), "afternoon")
         .when(F.col("hour").between(17, 20), "evening")
         .otherwise("night")
    )

    df = df.withColumn(
        "is_weekend",
        F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0)
    )

    return df


def create_derived_features(df):
    if "surge_multiplier" in df.columns:
        df = df.withColumn(
            "is_surge",
            F.when(F.col("surge_multiplier") > 1.0, 1).otherwise(0)
        )

    if "distance" in df.columns:
        df = df.withColumn(
            "distance_category",
            F.when(F.col("distance") < 1, "short")
             .when(F.col("distance") < 3, "medium")
             .otherwise("long")
        )

    return df


# =============================================================================
# 4. PREPROCESSING PIPELINE
# =============================================================================

def create_preprocessing_pipeline(categorical_cols, numeric_cols):
    stages = []

    index_cols = []
    for col in categorical_cols:
        indexer = StringIndexer(
            inputCol=col,
            outputCol=f"{col}_idx",
            handleInvalid="keep"
        )
        stages.append(indexer)
        index_cols.append(f"{col}_idx")

    encoded_cols = []
    for col in index_cols:
        encoder = OneHotEncoder(
            inputCol=col,
            outputCol=f"{col}_vec"
        )
        stages.append(encoder)
        encoded_cols.append(f"{col}_vec")

    assembler = VectorAssembler(
        inputCols=encoded_cols + numeric_cols,
        outputCol="features_raw",
        handleInvalid="skip"
    )

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    stages.extend([assembler, scaler])

    return Pipeline(stages=stages)


# =============================================================================
# 5. MODEL TRAINING
# =============================================================================

def train_linear_regression(train_data):
    print("\n🚀 Training Linear Regression Model")

    lr = LinearRegression(
        featuresCol="features",
        labelCol="price",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.5
    )

    model = lr.fit(train_data)

    print("✅ Linear Regression Trained")
    return model


# =============================================================================
# 6. MODEL EVALUATION
# =============================================================================

def evaluate_model(model, test_data):
    predictions = model.transform(test_data)

    evaluator_rmse = RegressionEvaluator(
        labelCol="price",
        predictionCol="prediction",
        metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol="price",
        predictionCol="prediction",
        metricName="mae"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol="price",
        predictionCol="prediction",
        metricName="r2"
    )

    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    print("\n📊 Model Performance")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"R²   : {r2:.4f}")

    predictions.select("price", "prediction").show(10)

    return predictions


# =============================================================================
# 7. SAVE / LOAD MODELS
# =============================================================================

def save_model(model, path):
    model.write().overwrite().save(path)
    print(f"💾 Model saved at {path}")


def load_model(path):
    return PipelineModel.load(path)


# =============================================================================
# 8. MAIN PIPELINE
# =============================================================================

def main():
    spark = create_spark_session()

    # Load data
    df = load_data(spark, "hdfs://localhost:9000/uber_data/cab_rides.csv")

    # Clean data
    df = df.filter(F.col("price").isNotNull()) \
           .filter(F.col("price") > 0) \
           .filter(F.col("distance") > 0)

    # Feature engineering
    df = extract_datetime_features(df)
    df = create_derived_features(df)

    # Feature selection
    categorical_cols = [
        "cab_type",
        "name",
        "source",
        "destination",
        "time_of_day",
        "distance_category"
    ]

    numeric_cols = [
        "distance",
        "surge_multiplier",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_surge"
    ]

    categorical_cols = [c for c in categorical_cols if c in df.columns]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Preprocessing
    pipeline = create_preprocessing_pipeline(categorical_cols, numeric_cols)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    pipeline_model = pipeline.fit(train_df)

    train_transformed = pipeline_model.transform(train_df)
    test_transformed = pipeline_model.transform(test_df)

    # Train model
    lr_model = train_linear_regression(train_transformed)

    # Evaluate
    evaluate_model(lr_model, test_transformed)

    # Save models
    save_model(pipeline_model, "models/preprocessing_pipeline")
    save_model(lr_model, "models/linear_regression_model")

    spark.stop()
    print("\n✅ PIPELINE EXECUTED SUCCESSFULLY")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()