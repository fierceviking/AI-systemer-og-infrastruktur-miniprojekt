import warnings
import logging
import os
import pyspark
import findspark
from pyspark.sql import SparkSession
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set logging level to ERROR to ignore warnings
logging.getLogger("py4j").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

findspark.init()
spark = SparkSession.builder.appName("pizza_sales").getOrCreate()

def load_data(file_name):
    # Define data path
    data = os.path.join(os.path.dirname(__file__), file_name)

    # Read data
    df_spark = spark.read.csv(data, header=True, inferSchema=True)
    return df_spark


def correlation_matrix(df):
    correlation = df.corr(numeric_only=True)
    ax = sns.heatmap(correlation, annot=True)
    plt.show()

def main():
    df_spark = load_data('pizza_sales.csv')
    print(df_spark.printSchema())
    df_spark = df_spark.drop('order_date', 'order_time','pizza_id','order_id')
    # df_spark.drop(['order_date','order_time'])
    df_pd = df_spark.toPandas()
    correlation_matrix(df_pd)

if __name__ == '__main__':
    main()
