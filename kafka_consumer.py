from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from pyspark.ml.regression import LinearRegressionModel

# ----------------------------------
# Spark Session
# ----------------------------------
spark = (
    SparkSession.builder
    .appName("CabPriceKafkaPrediction")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# ----------------------------------
# Load trained models
# ----------------------------------
pipeline_model = PipelineModel.load("models/preprocessing_pipeline")
lr_model = LinearRegressionModel.load("models/linear_regression_model")

# ----------------------------------
# Kafka Schema
# ----------------------------------
schema = StructType([
    StructField("cab_type", StringType()),
    StructField("name", StringType()),
    StructField("source", StringType()),
    StructField("destination", StringType()),
    StructField("distance", DoubleType()),
    StructField("surge_multiplier", DoubleType()),
    StructField("hour", IntegerType()),
    StructField("day_of_week", IntegerType()),
    StructField("month", IntegerType())
])

# ----------------------------------
# Read Kafka Stream
# ----------------------------------
kafka_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "cab_price_features")
    .option("startingOffsets", "latest")
    .load()
)

# ----------------------------------
# Parse JSON
# ----------------------------------
parsed_df = (
    kafka_df
    .selectExpr("CAST(value AS STRING)")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
)

# ----------------------------------
# Feature Engineering (same as training)
# ----------------------------------
features_df = (
    parsed_df
    .withColumn(
        "time_of_day",
        when(col("hour").between(6,11),"morning")
        .when(col("hour").between(12,16),"afternoon")
        .when(col("hour").between(17,20),"evening")
        .otherwise("night")
    )
    .withColumn("is_weekend", when(col("day_of_week").isin(1,7),1).otherwise(0))
    .withColumn("is_surge", when(col("surge_multiplier") > 1.0,1).otherwise(0))
    .withColumn(
        "distance_category",
        when(col("distance") < 1,"short")
        .when(col("distance") < 3,"medium")
        .otherwise("long")
    )
)

# ----------------------------------
# Transform + Predict
# ----------------------------------
transformed = pipeline_model.transform(features_df)
predictions = lr_model.transform(transformed)

# ----------------------------------
# Select Output
# ----------------------------------
output_df = predictions.select(
    "cab_type",
    "name",
    "distance",
    "surge_multiplier",
    col("prediction").alias("predicted_price")
)

# ----------------------------------
# Write to TERMINAL
# ----------------------------------
query = (
    output_df
    .writeStream
    .format("console")
    .outputMode("append")
    .option("truncate", False)
    .start()
)

query.awaitTermination()
