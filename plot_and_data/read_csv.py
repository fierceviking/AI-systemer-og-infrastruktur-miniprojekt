import warnings
warnings.filterwarnings("ignore")

import logging

# Set logging level to ERROR to ignore warnings
logging.getLogger("py4j").setLevel(logging.ERROR)

import os
import pyspark
import findspark
from pyspark.sql import SparkSession

findspark.init()
spark = SparkSession.builder.appName("pizza_sales").getOrCreate()


data = os.path.join(os.path.dirname(__file__), 'pizza_sales.csv')

df_spark = spark.read.csv(data, header=True, inferSchema=True)

print(df_spark.head(5))