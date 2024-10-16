from pie_chart import load_data
import warnings
import logging
import os
import pyspark
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, count, isnan, split, explode
import pandas as pd
import matplotlib.pyplot as plt


def bar_chart(df, feature):
    # Split the specified column
    df_split = df.withColumn(feature, split(col(feature), ', '))
    # df_split.select('pizza_ingredients').show()

    # Make each ingredient have its own row
    df_exploded = df_split.withColumn(feature, explode(col(feature)))

    df_feature = df_exploded.groupBy(feature).agg(sum('quantity').alias('quantity'))
    # df_feature.show(5)

    # Convert to pandas DF
    df_pd = df_feature.toPandas()
    print(df_pd.head(3))
    print(df_pd.shape)

    # Plot
    

    plt.figure(figsize=(8,8))
    plt.bar(x=df_pd.loc[:,feature], height=df_pd.loc[:,'quantity'])
    plt.title(f"Bar chart of {feature}")
    plt.legend()
    # plt.show()

def main():
    df = load_data('pizza_sales.csv')
    # print(df.columns)

    bar_chart(df, 'pizza_ingredients')

if __name__ == '__main__':
    main()