import warnings
import logging
import os
import pyspark
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, count, isnan, lit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    # Use Pyspark SQL to show the pizza_sizes and their respective quantity
    df_pizza = df.groupBy("pizza_size").agg(sum("quantity").alias("quantity"))

    df_combined = df_pizza.filter(df_pizza['pizza_size'].isin('XL', 'XXL')) \
                    .agg(lit('XL + XXL').alias('pizza_size'), sum('quantity').alias('quantity'))
    
    # df_combined.show()

    # Combine the two dataframes
    df_result = df_pizza.union(df_combined)
    df_result = df_result.filter(df_result['pizza_size'].isin('S','M','L','XL + XXL'))

    df_result.show()

    # Convert to pandas DF
    df_pd = df_result.toPandas()
    # print(df_pd.head(3))

    # Visualize using matplotlib
    explode = (0.1, 0, 0, 0)
    color_palette = sns.color_palette("YlGnBu")
    plt.figure(figsize=(8,8))
    plt.pie(
        x=df_pd['quantity'],
        labels=df_pd['pizza_size'], 
        autopct='%1.1f%%', 
        explode=explode,
        colors=color_palette,
        shadow=True,
        textprops={'fontsize': 20})
    plt.title('Distribution of Sales pr. Pizza Sizes')
    plt.axis('equal')
    plt.legend()
    plt.show()

def pie_chart_pizzas(df, pizza_size):
    # Filter the data by the pizza size
    df_pizza = df.filter(col('pizza_size')==pizza_size) 
    # df_pizza.show()

    # Groupby the pizza_category column and count the occurences
    df_grouped = df_pizza.groupBy("pizza_category").count()
    # df_grouped.show()

    # Convert to pandas DF
    df_pd = df_grouped.toPandas()
    # print(df_pd.head(3))

    color_palette = sns.color_palette("YlGnBu")
    # Plot the pie chart
    plt.figure(figsize=(8,8))
    plt.pie(
        x=df_pd['count'],
        labels=df_pd['pizza_category'], 
        autopct='%1.1f%%',
        colors=color_palette
        )
    plt.title(f'Pie chart of pizza names for pizza size {pizza_size}')
    plt.axis('equal')
    plt.legend()
    plt.show()



def main():
    # Load the data
    df = load_data('../pizza_sales.csv')
    # df.show(5, truncate=True) # Show the first 5 rows
    
    # Data insights
    data_description(df)

    # # Missing values
    # df_filtered = missing_values(df)
    # df_filtered.show()

    # Pie chart
    # pie_chart(df)
    
    pie_chart_pizzas(df, 'S')

if __name__ == '__main__':
    main()