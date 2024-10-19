import warnings
warnings.filterwarnings("ignore")

import os
import datetime
import pyspark
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


# Code from Haomin start -->
# Cast 'order_time' from string to timestamp
exc1_df = exc1_df.withColumn("order_time", col("order_time").cast("string"))
exc1_df = exc1_df.withColumn("order_time", substring(col("order_time"), 12, 8))
exc1_df = exc1_df.withColumn(
    'order_datetime',
    concat(col('order_date'), lit(' '), col('order_time'))
)
 
 
exc1_df = exc1_df.withColumn(
    'day_of_week',
    dayofweek(to_timestamp(col('order_datetime'), 'M/d/yyyy HH:mm:ss'))  # Specify the format
)

# Code from Haomin end <--

exc1_df = exc1_df.drop('order_datetime', 'order_date')

exc1_df.show()

print(exc1_df.schema)

exc1_data = exc1_df.collect()
exc1_cols = exc1_df.columns

exc1_df = pd.DataFrame(exc1_data, columns=exc1_cols)

print(exc1_df['day_of_week'].unique())

exc1_df['order_time'] = pd.to_datetime(exc1_df['order_time'], format='%H:%M:%S')

exc1_df['day_of_week'] = exc1_df['day_of_week'] - 1
exc1_df['hour'] = exc1_df['order_time'].dt.hour

aggregated_df = exc1_df.groupby(['day_of_week', 'hour'], as_index=False)['quantity'].sum()

plt.figure(figsize=(12, 6))

sns.lineplot(data=aggregated_df, x='hour', y='quantity', hue='day_of_week', marker='o', palette='tab10')

plt.title('Pizza Orders per Hour Colored by Day of the Week')
plt.xlabel('Hour of Day')
plt.ylabel('Quantity of Pizzas')
plt.xticks(range(9, 25))
plt.legend(title='Day of Week', labels=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.grid(True)

plt.show()