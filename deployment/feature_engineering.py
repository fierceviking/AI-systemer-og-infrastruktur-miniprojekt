import pyspark
import os
import warnings
import findspark
from pyspark.sql import SparkSession, Window
import logging
from pyspark.sql.types import DecimalType
from pyspark.sql.functions import col, substring, concat, lit, to_timestamp, \
    month, sum, hour, dayofmonth, when, to_date, date_trunc, \
    lag, regexp_replace, dayofweek

# Filter warnings
warnings.filterwarnings("ignore")
logging.getLogger("py4j").setLevel(logging.ERROR)
findspark.init()
spark = SparkSession.builder.appName("pizza_sales").getOrCreate()

def load_data(file_name):
    # Define data path
    data = os.path.join(os.path.dirname(__file__), file_name)

    # Read data
    df_spark = spark.read.csv(data, header=True, inferSchema=True)
    return df_spark

def convert_to_datetime(exc1_df):
    """
    This function adds month, day_of_week, and hour columns to the DataFrame based on 'order_date' and 'order_time'.
    """
    # Select relevant columns
  #  exc1_df = df_spark.select("order_date", "order_time", "quantity", "total_price")

    # DecimalType ensures that pyspark can sum quantity and total_price
    # (10, 2) ensures that pyspark can sum with 2 decimals and 10 digits total
    # FloatType can give floating-point precision errors
    exc1_df = exc1_df.withColumn("quantity", exc1_df["quantity"].cast(DecimalType()))
    exc1_df = exc1_df.withColumn("total_price", exc1_df["total_price"].cast(DecimalType(10, 2)))

    # Standardize 'order_date' format
    exc1_df = exc1_df.withColumn("order_date", regexp_replace(col("order_date"), "-", "/"))

    # Extract time substring and create 'order_timestamp' column
    exc1_df = exc1_df.withColumn("order_time", substring(col("order_time"), 12, 8))
    exc1_df = exc1_df.withColumn(
        "order_timestamp",
        to_timestamp(concat(col("order_date"), lit(" "), col("order_time")), "d/M/yyyy HH:mm:ss")
    )

    # Add month, day_of_week, and hour columns
    exc1_df = exc1_df.withColumn("day_of_week", dayofweek(col("order_timestamp")))
    exc1_df = exc1_df.withColumn("month", month(col("order_timestamp")))
    exc1_df = exc1_df.withColumn("hour", hour(col("order_timestamp")))

    return exc1_df

def create_dataframe(df_spark):
    # Round the hour in timestamp-feature
    df_spark = df_spark.withColumn('order_timestamp_hour', date_trunc('hour', col('order_timestamp')))

    # Calculate quantity of each pizza size per hour
    df_spark = df_spark.groupBy(
        'order_timestamp_hour', 'day_of_week','month', 'hour'
        ).agg(
        sum(when(col("pizza_size") == "S", col("quantity")).otherwise(0)).alias("S_count"),
        sum(when(col("pizza_size") == "M", col("quantity")).otherwise(0)).alias("M_count"),
        sum(when(col("pizza_size") == "L", col("quantity")).otherwise(0)).alias("L_count"),
        sum(when(col("pizza_size") == "XL", col("quantity")).otherwise(0)).alias("XL_count"),
        sum(when(col("pizza_size") == "XXL", col("quantity")).otherwise(0)).alias("XXL_count"),
        sum(when(col("pizza_category") == 'Classic', col("quantity")).otherwise(0)).alias('Classic'),
        sum(when(col("pizza_category") == 'Chicken', col("quantity")).otherwise(0)).alias('Chicken'),
        sum(when(col("pizza_category") == 'Supreme', col("quantity")).otherwise(0)).alias('Supreme'),
        sum(when(col("pizza_category") == 'Veggie', col("quantity")).otherwise(0)).alias('Veggie'),
        sum(col("quantity")).alias("total_quantity"),  # Total quantity of pizzas sold per hour
        sum('total_price').alias('total_sales') # Total sales pr hour
    ).orderBy('order_timestamp_hour', 'hour')

    # Sort dataframe
    df_spark = df_spark.sort('order_timestamp_hour', 'hour')
    df_spark = df_spark.filter(col("day_of_week").isNotNull())

    return df_spark

def lag_variable(df_spark, feature, offset):
    # Define a window specification with better partitioning and ordering
    window_spec = Window.partitionBy('month').orderBy('order_timestamp_hour')
    
    # Apply the lag function
    df_spark = df_spark.withColumn(f'lag_{feature}_{offset}', lag(feature, offset=offset).over(window_spec))
    
    # Replace missing values with 0
    df_spark = df_spark.fillna({f'lag_{feature}_{offset}': 0})
    
    return df_spark

def main():
    df = load_data('../pizza_sales.csv')
    # df.show(10)

    df_hour = convert_to_datetime(df)
    # df_hour.show(5)

    df_featured = create_dataframe(df_hour)

    features = ['total_quantity', 'total_sales', 'S_count', 'M_count', 'L_count', 'XL_count', 'XXL_count', 'Classic', 'Chicken', 'Supreme', 'Veggie']
    for feature in features:

        df_featured = lag_variable(df_featured, feature, offset=1)
        df_featured = lag_variable(df_featured, feature, offset=3)
        df_featured = lag_variable(df_featured, feature, offset=5)

    df_featured.show(20)

    df_featured.toPandas().to_csv('new_pizza_sales.csv')

if __name__ == '__main__':
    main()