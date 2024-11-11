import findspark
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import DecimalType
from pyspark.sql.functions import sum, col
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import logging

def fliter_warning():
    # Filter warnings and initialize spark in environment
    warnings.filterwarnings("ignore")
    logging.getLogger("py4j").setLevel(logging.ERROR)
    findspark.init()
    spark = SparkSession.builder.appName("pizza_sales").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_data(file_name, spark):
    # Defines the path to the csv file
    data = os.path.join(os.path.dirname(__file__), file_name)

    df_spark = spark.read.csv(data, header=True, inferSchema=True)
    return df_spark

def pivot_table(df):
    # DecimalType ensures that pyspark can sum quantity and total_price
    # (10, 2) ensures that pyspark can sum with 2 decimals and 10 digits total
    # FloatType can give floating-point precision errors
    df = df.withColumn("quantity", df["quantity"].cast(DecimalType()))
    df = df.withColumn("total_price", df["total_price"].cast(DecimalType(10, 2)))

    # Sums x in a list with one row and one field
    # [0][0] indicates first row and first field in that row
    total_revenue = df.agg(sum("total_price")).collect()[0][0]
    total_sale = df.agg(sum("quantity")).collect()[0][0]

    # Creates the pyspark pivot table
    df_pivot_spark = df.groupBy("pizza_category", "pizza_size") \
        .agg(sum("total_price").alias("Total Income"), 
            (sum("total_price") / total_revenue * 100).alias("Income in Percentage"), 
            sum("quantity").alias("Quantity of Pizzas"), 
            (sum("quantity") / total_sale * 100).alias("Quantities in Percentage"))\
        .orderBy(col("Total Income").desc(), col("Quantity of Pizzas").desc())
    
    # Rename the groupBy columns
    df_pivot_spark = df_pivot_spark.withColumnRenamed("pizza_category", "Pizza Category") \
                                   .withColumnRenamed("pizza_size", "Pizza Size")
   
   # df_order_count.show()
    return df_pivot_spark

def better_looking_pivot_table(df_pivot_spark):
    
    # Converst to Pandas
    df_pivot_pandas = df_pivot_spark.toPandas()

    fig, ax = plt.subplots(figsize=(12, 6)) # Creates the subplot
    ax.axis('tight') # Makes the layout fit tightly around the table
    ax.axis('off') # Turns the extra lines and borders off

    # Creates the table, centers everything, turns scaling off, and sets font size
    table = ax.table(cellText=df_pivot_pandas.values, colLabels=df_pivot_pandas.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df_pivot_pandas.columns))))

    plt.show()
    
def main():
    file_name = '../pizza_sales.csv'
    spark = fliter_warning()
    df_spark = load_data(file_name, spark)

    df_pivot_spark = pivot_table(df_spark)

    better_looking_pivot_table(df_pivot_spark)
    
    spark.stop() # Stops the pyspark session

if __name__ == '__main__':
    main()