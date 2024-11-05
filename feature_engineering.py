import pyspark
import os
import warnings
import findspark
from pyspark.sql import SparkSession
import logging
from pyspark.sql.functions import col, substring, concat, lit, to_timestamp, month, sum, hour

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

def convert_to_datetime(df_spark):
    exc1_df = df_spark.select("order_date", "order_time", "pizza_size", 'quantity', 'total_price')
    exc1_df = exc1_df.withColumn("quantity", col("quantity").cast("int"))
    exc1_df = exc1_df.withColumn("total_price", col("total_price").cast("float"))

    # Ensure 'order_date' and 'order_time' are strings
    exc1_df = exc1_df.withColumn("order_date", col("order_date").cast("string"))
    exc1_df = exc1_df.withColumn("order_time", col("order_time").cast("string"))
    
    # Extract time substring and create 'order_datetime'
    exc1_df = exc1_df.withColumn("order_time", substring(col("order_time"), 12, 8))
    exc1_df = exc1_df.withColumn(
        'order_datetime',
        concat(col('order_date'), lit(' '), col('order_time'))
    )

    # Convert to timestamp and extract month and hour
    exc1_df = exc1_df.withColumn(
        'order_timestamp',
        to_timestamp(col('order_datetime'), 'M/d/yyyy HH:mm:ss')
    )
    exc1_df = exc1_df.withColumn('month', month(col('order_timestamp')))
    exc1_df = exc1_df.withColumn('hour', hour(col('order_timestamp')))

    # Drop intermediate columns
    exc1_df = exc1_df.drop('order_datetime', 'order_date', 'order_time', 'order_timestamp')
    
    # Group by 'month', 'pizza_size', and 'hour' and aggregate total_price
    hourly_sales = exc1_df.groupBy('month', 'pizza_size', 'hour').agg(
        sum('total_price').alias('total_price')
    ).orderBy('month', 'pizza_size', 'hour')

    # Filter out rows with null month values
    hourly_sales = hourly_sales.filter(col("month").isNotNull())
    
    return hourly_sales

def main():
    df = load_data('pizza_sales.csv')
    # df.show(10)

    df_hour = convert_to_datetime(df)
    df_hour.show(5)

if __name__ == '__main__':
    main()