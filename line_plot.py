from pizza_pivot import fliter_warning, load_data

import os
import datetime
import pyspark
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, DecimalType
from pyspark.sql.functions import col
from pyspark.sql.functions import substring
from pyspark.sql.functions import concat
from pyspark.sql.functions import lit
from pyspark.sql.functions import dayofweek
from pyspark.sql.functions import to_timestamp

def edit_data(df_spark, size_or_category):
    df_spark = df_spark.withColumn('quantity', df_spark['quantity'].cast(DecimalType()))

    # 1. How can we compare the sales across 24 hours of the day? (Bar chart)
    exc1_df = df_spark.select(size_or_category, 'order_time', 'quantity')

    # Here we collect the data and columns to convert the pyspark dataframe to a pandas dataframe
    exc1_data = exc1_df.collect()
    exc1_cols = exc1_df.columns

    exc1_df = pd.DataFrame(exc1_data, columns=exc1_cols) # The pandas dataframe is created

    # Here the in-built function in pandas is used to convert the order time to the datetime format. 
    # It adds a date, but it doesn't matter since the hour is the only relevant data
    exc1_df['order_time'] = pd.to_datetime(exc1_df['order_time'], format='%H:%M:%S')

    # Here the hour is extracted from the order_time column
    exc1_df['hour'] = exc1_df['order_time'].dt.hour

    # Here the quanity is aggregated around day_of_week and hour. 
    # This is then summed to get the amount of pizza sales per hour for each weekday
    aggregated_df = exc1_df.groupby([size_or_category, 'hour'], as_index=False)['quantity'].sum()
    return aggregated_df

def line_plot(df_edited, size_or_category):
    # The figure size is defined here
    plt.figure(figsize=(12, 6))

    # The plot is created with seaborn
    sns.lineplot(data=df_edited, x='hour', y='quantity', hue=size_or_category, marker='o', palette='tab10')

    # Better name
    if size_or_category == 'pizza_size':
        size_or_category = 'Pizza Size'
    else:
        size_or_category = 'Pizza Category'

    # Boiler plate figure code
    plt.title(f'Pizza Orders per Hour Colored by {size_or_category}', fontsize=20)
    plt.xlabel('Hour of Day', fontsize=20)
    plt.ylabel('Quantity of Pizzas', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(range(9, 25), fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)

    plt.show()


def main():
    file_name = 'pizza_sales.csv'
    size_or_category = ['pizza_size', 'pizza_category']

    spark = fliter_warning()
    df_spark = load_data(file_name, spark)

    for type in size_or_category:
        df_edited = edit_data(df_spark, type)
        line_plot(df_edited, type)
    
    spark.stop() # Stops the pyspark session


if __name__ == '__main__':
    main()

# DISCLAIMER: Notes haven't been edited. May be incorrect
