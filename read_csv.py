import warnings
warnings.filterwarnings("ignore")

import logging

# Set logging level to ERROR to ignore warnings
logging.getLogger("py4j").setLevel(logging.ERROR)

import os
import datetime
import pyspark
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf

findspark.init()
spark = SparkSession.builder.appName("pizza_sales").getOrCreate()


data = os.path.join(os.path.dirname(__file__), 'pizza_sales.csv')

df_spark = spark.read.csv(data, header=True, inferSchema=True)

print(df_spark.show())

# 1. How can we compare the sales across 24 hours of the day? (Bar chart)

def toWeekday(date):
    try:
        date = date.split(" ") # This seperates the date and time
        print(date)
        date = date[0].split("-") # This seperates the components of the date
        print(date)
        day = datetime.datetime(int(date[0]), int(date[1]), int(date[2]))
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[day.weekday()]
    except TypeError: # This should ensure that no errors arise
        return None

weekdayFunc = udf(toWeekday)

exc1 = df_spark.select("order_time")
weekdayColumn = exc1.foreach(datetime.date.weekday)
print(weekdayColumn)
exc1.show()