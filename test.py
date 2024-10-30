from pizza_pivot import load_data
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

def pie_chart(df):
    # Use Pyspark SQL to show the pizza_sizes and their respective quantity
    df_pizza = df.groupBy("pizza_size").agg(sum("quantity").alias("quantity"))

  #  df_combined = df_pizza.filter(df_pizza['pizza_size'].isin('XL', 'XXL')) \
   #                 .agg(lit('XL + XXL').alias('pizza_size'), sum('quantity').alias('quantity'))
    
    # df_combined.show()

    # Combine the two dataframes
   # df_result = df_pizza.union(df_pizza)
  #  df_result = df_result.filter(df_result['pizza_size'].isin('S','M','L','XL + XXL'))

   # df_result.show()

    # Convert to pandas DF
    df_pd = df_pizza.toPandas()
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

def pie_chart_pizzas(df):

    # Groupby the pizza_category column and count the occurences
    df_grouped = df.groupBy("pizza_category").count()
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
    plt.title(f'Pie chart of pizza names for pizza size ')
    plt.axis('equal')
    plt.legend()
    plt.show()



def main():
    # Load the data
    df = load_data('pizza_sales.csv')

    # Pie chart
    pie_chart(df)
    
   # pie_chart_pizzas(df)

if __name__ == '__main__':
    main()