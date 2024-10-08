import warnings
import logging
import os
import pyspark
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, count, isnan
import pandas as pd
import matplotlib.pyplot as plt


# Filter warnings
warnings.filterwarnings("ignore")
logging.getLogger("py4j").setLevel(logging.ERROR)
findspark.init()
spark = SparkSession.builder.appName("pizza_sales").getOrCreate()

# # Adjust maxToStringFields
# spark.conf.set("spark.sql.debug.maxToStringFields", 10)  # You can set any value you want
# print(spark.conf.get("spark.sql.debug.maxToStringFields"))


def load_data(file_name):
    # Define data path
    data = os.path.join(os.path.dirname(__file__), file_name)

    # Read data
    df_spark = spark.read.csv(data, header=True, inferSchema=True)
    return df_spark

def data_description(df):
    print('Columns names:')
    df.columns

    print('\nData types:')
    df.printSchema()

    print('\nData summary:')
    df.summary().show()

def missing_values(df):
    df = df.drop('order_time')
    df_filtered = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) 
                             for c in df.columns])
    return df_filtered

def pie_chart(df):
    # df.select("pizza_size").distinct().show() # Output: S, M, L, XL, XXL

    # Get a list of the unique values in pizza_size column
    unique_sizes = [row['pizza_size'] for row in df.select("pizza_size").distinct().collect()]
    
    # Create dictionary with sizes as key and sum of quntity as value
    pizza_dictionary = {'pizza_sizes': [],
                        'quantity': []}
    for size in unique_sizes:
        quantity_sum = df.filter(col('pizza_size')==size).agg(sum("quantity")).collect()[0][0]
        pizza_dictionary['pizza_sizes'].append(str(size))
        pizza_dictionary['quantity'].append(int(quantity_sum))
    # print(pizza_dictionary)

    # Make a new PySpark dataframe based on the dictionary
    data_tuples = list(zip(pizza_dictionary['pizza_sizes'], pizza_dictionary['quantity']))
    df_pizza = spark.createDataFrame(data_tuples, schema=["pizza_sizes", "quantity"])
    # df_pizza.show()

    # Convert PySpark df to pandas df
    df_pd = df_pizza.toPandas()
    print(df_pd)

    # Visualize using matplotlib
    explode = (0.1, 0, 0, 0, 0.2)
    plt.figure(figsize=(8,8))
    plt.pie(
        x=df_pd['quantity'],
        labels=df_pd['pizza_sizes'], 
        autopct='%1.1f%%', 
        explode=explode)
    plt.title('Pie chart of quantity pr. pizza size')
    plt.axis('equal')
    plt.legend()
    plt.show()

def main():
    # Load the data
    df = load_data('pizza_sales.csv')
    # df.show(5, truncate=True) # Show the first 5 rows
    
    # Data insights
    # data_description(df)

    # # Missing values
    df_filtered = missing_values(df)
    df_filtered.show()

    # Pie chart
    pie_chart(df)
    

if __name__ == '__main__':
    main()