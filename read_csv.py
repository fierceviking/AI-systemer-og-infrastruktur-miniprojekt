import warnings
warnings.filterwarnings("ignore")

import os
import datetime
import pyspark
import pandas
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.sql.functions import substring
from pyspark.sql.functions import concat
from pyspark.sql.functions import lit
from pyspark.sql.functions import dayofweek
from pyspark.sql.functions import to_timestamp

findspark.init()
spark = SparkSession.builder.appName("pizza_sales").getOrCreate()


data = os.path.join(os.path.dirname(__file__), 'pizza_sales.csv')

df_spark = spark.read.csv(data, header=True, inferSchema=True)

# 1. How can we compare the sales across 24 hours of the day? (Bar chart)
exc1_df = df_spark.select("order_date", "order_time", "quantity")
 
 
columns = ["order_time"]
# df_spark_tmp = spark.createDataFrame(data, columns)
 
# Cast 'order_time' from string to timestamp
df_spark_tmp = df_spark.withColumn("order_time", col("order_time").cast("string"))
df_spark_tmp = df_spark_tmp.withColumn("order_time", substring(col("order_time"), 12, 8))
df_spark_tmp = df_spark_tmp.withColumn(
    'order_datetime',
    concat(col('order_date'), lit(' '), col('order_time'))
)
 
 
df_spark_tmp = df_spark_tmp.withColumn(
    'day_of_week',
    dayofweek(to_timestamp(col('order_datetime'), 'M/d/yyyy HH:mm:ss'))  # Specify the format
)

pandas_df = df_spark_tmp.toPandas()
print(pandas_df['day_of_week'].unique())

pandas_df[pandas_df["day_of_week"] == 1]