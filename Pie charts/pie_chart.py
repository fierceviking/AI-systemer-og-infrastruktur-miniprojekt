import warnings
import logging
import os
import pyspark
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, count, isnan, lit
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
logging.getLogger("py4j").setLevel(logging.ERROR)
findspark.init()
spark = SparkSession.builder.appName("pizza_sales").getOrCreate()

def load_data(file_name):
    data = os.path.join(os.path.dirname(__file__), file_name)
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

    # Combine the two dataframes
    df_result = df_pizza.union(df_combined)
    df_result = df_result.filter(df_result['pizza_size'].isin('S','M','L','XL + XXL'))

    df_pd = df_result.toPandas()

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

    # Groupby the pizza_category column and count the occurences
    df_grouped = df_pizza.groupBy("pizza_category").count()

    df_pd = df_grouped.toPandas()

    color_palette = sns.color_palette("YlGnBu")

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
    df = load_data('../pizza_sales.csv')

    # Data insights
    data_description(df)
    pie_chart_pizzas(df, 'S')

if __name__ == '__main__':
    main()