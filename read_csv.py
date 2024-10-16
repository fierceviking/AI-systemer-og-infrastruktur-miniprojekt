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

findspark.init()
spark = SparkSession.builder.appName("pizza_sales").getOrCreate()


data = os.path.join(os.path.dirname(__file__), 'pizza_sales.csv')

df_spark = spark.read.csv(data, header=True, inferSchema=True)

# 1. How can we compare the sales across 24 hours of the day? (Bar chart)

exc1_df = df_spark.select("order_date", "order_time", "quantity")

def toWeekday(date):
        date = str(date).split("/") # This separates the date and time
        day, month, year = int(date[0]), int(date[1]), int(date[2])
        day = datetime.datetime(year, month, day)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[day.weekday()]

def formatTime(time):
    time = str(time).split(" ")
    print(time)
    return time[1]

weekdayConv = udf(toWeekday, StringType())

timeConv = udf(formatTime, StringType())

exc1_df = exc1_df.withColumn("weekday", weekdayConv(exc1_df["order_date"]))

exc1_df = exc1_df.withColumn("order_time", timeConv(exc1_df["order_time"]))

# exc1_pandas = exc1_df.toPandas()