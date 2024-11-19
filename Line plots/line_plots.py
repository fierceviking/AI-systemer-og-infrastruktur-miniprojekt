import warnings
warnings.filterwarnings("ignore")

import os
import datetime
import pyspark
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
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
from pyspark.sql.functions import month

def hourly_sales(pyspark_dataframe):
    # 1. How can we compare the sales across 24 hours of the day? (Bar chart)
    exc1_df = pyspark_dataframe.select("order_date", "order_time", "quantity")


    # Code from Haomin Yu start -->
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
    # Code from Haomin Yu end <--

    exc1_df = exc1_df.drop('order_datetime', 'order_date') # Here we drop the unwanted columns that were used for order_datetime

    # Here we collect the data and columns to convert the pyspark dataframe to a pandas dataframe
    exc1_data = exc1_df.collect()
    exc1_cols = exc1_df.columns

    exc1_df = pd.DataFrame(exc1_data, columns=exc1_cols) # The pandas dataframe is created

    # Here the in-built function in pandas is used to convert the order time to the datetime format. 
    # It adds a date, but it doesn't matter since the hour is the only relevant data
    exc1_df['order_time'] = pd.to_datetime(exc1_df['order_time'], format='%H:%M:%S')

    # Since the code from Haomin gives weekdays 1-7, we subtract 1 to make it zero-indexed (see line 89)
    exc1_df['day_of_week'] = exc1_df['day_of_week'] - 1

    # Here the hour is extracted from the order_time column
    exc1_df['hour'] = exc1_df['order_time'].dt.hour

    # Here the quanity is aggregated around day_of_week and hour. 
    # This is then summed to get the amount of pizza sales per hour for each weekday
    aggregated_df = exc1_df.groupby(['day_of_week', 'hour'], as_index=False)['quantity'].sum()

    weekdays = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}

    aggregated_df['day_of_week'] = aggregated_df['day_of_week'].map(weekdays)

    # The figure size is defined here
    plt.figure(figsize=(12, 4))

    # Seaborn plot
    sns.lineplot(data=aggregated_df, x='hour', y='quantity', hue='day_of_week', marker='o', palette='tab10')

    # Boiler plate figure code
    plt.title('Pizza Orders per Hour Colored by Day of the Week', fontsize=20)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xticks(range(9, 25), fontsize=20)
    plt.yticks([0, 100, 200, 300, 400], fontsize=20)

    # Allow Seaborn to automatically assign colors in the legend
    plt.legend(title='Day of Week', fontsize=14)
    plt.grid(True)

    plt.show()
    return aggregated_df

def daily_sales(spark_dataframe):
    # Plotting sales per day
    exc2_df = spark_dataframe.select('order_date', 'quantity')

    exc2_df = exc2_df.groupBy('order_date').agg({'quantity': 'sum'})

    exc2_data = exc2_df.collect()
    exc2_cols = exc2_df.columns

    exc2_pd = pd.DataFrame(data = exc2_data, columns = exc2_cols)

    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(12,6))
    sns.lineplot(exc2_pd, x='order_date', y='sum(quantity)')
    plt.gca().xaxis.set_major_locator(mdate.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdate.DateFormatter('%d/%m'))
    plt.legend(title='Pizza sales per day')
    plt.grid(True)
    plt.show()
    return exc2_pd

def monthly_sales(spark_dataframe):
    """
    Plots sales per month. Takes a spark dataframe with some order date and sales quantity.
    """

    exc3_df = spark_dataframe.select('order_date', 'quantity')

    exc3_data = exc3_df.collect()
    exc3_cols = exc3_df.columns

    exc3_pd = pd.DataFrame(data = exc3_data, columns = exc3_cols)

    exc3_pd['order_date'] = pd.to_datetime(exc3_pd['order_date'], format='mixed', dayfirst=True)

    exc3_pd = exc3_pd.groupby(exc3_pd['order_date'].dt.month).agg({'quantity': 'sum'})

    print(exc3_pd.head(20))

    print(exc3_pd.columns)

    plt.figure(figsize=(12,6))
    sns.lineplot(exc3_pd, x = range(1, 13), y = exc3_pd['quantity'])
    plt.xticks(range(1, 13), fontsize=14)
    plt.legend(title='Pizza sales per month', fontsize=16)
    plt.grid(True)
    plt.show()
    return exc3_pd

def main():    
    findspark.init()
    spark = SparkSession.builder.appName("pizza_sales").getOrCreate()


    data = os.path.join(os.path.dirname(__file__), '../pizza_sales.csv')

    df_spark = spark.read.csv(data, header=True, inferSchema=True)

    hourly_sales(df_spark)

    daily_sales(df_spark)

    monthly_sales(df_spark)

    # dataframe = daily_sales(df_spark)

    # dataframe.to_csv("daily_sales.csv")

if __name__ == "__main__":
    main()