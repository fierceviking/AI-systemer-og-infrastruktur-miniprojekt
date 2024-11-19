import warnings
warnings.filterwarnings("ignore")
import os
import pyspark
import matplotlib.pyplot as plt
import seaborn as sns
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring, concat, lit, to_timestamp, month, sum

findspark.init()
spark = SparkSession.builder.appName("pizza_sales").getOrCreate()

def load_data(file_name):
    data = os.path.join(os.path.dirname(__file__), file_name)
    df_spark = spark.read.csv(data, header=True, inferSchema=True)
    return df_spark


def convert_to_datetime(df_spark):
    exc1_df = df_spark.select("order_date", "order_time", "pizza_size", 'quantity')
    exc1_df = exc1_df.withColumn("quantity", col("quantity").cast("int"))

    # Ensure 'order_date' and 'order_time' are strings
    exc1_df = exc1_df.withColumn("order_date", col("order_date").cast("string"))
    exc1_df = exc1_df.withColumn("order_time", col("order_time").cast("string"))
    
    # Extract time substring
    exc1_df = exc1_df.withColumn("order_time", substring(col("order_time"), 12, 8))

    # Concatenate to create 'order_datetime'
    exc1_df = exc1_df.withColumn(
        'order_datetime',
        concat(col('order_date'), lit(' '), col('order_time'))
    )

    # Convert to timestamp and extract month
    exc1_df = exc1_df.withColumn(
        'month',
        month(to_timestamp(col('order_datetime'), 'M/d/yyyy HH:mm:ss'))  # Specify the format
    )

    exc1_df = exc1_df.drop('order_datetime', 'order_date', 'order_time')  # Drop unwanted columns
    
    # Group by 'month' and 'pizza_size' and aggregate the quantity
    monthly_pizza = exc1_df.groupBy('month', 'pizza_size').agg(
        sum('quantity').alias('quantity')
    ).orderBy('month', 'pizza_size')

    # Filter out rows with null month values
    monthly_pizza = monthly_pizza.filter(col("month").isNotNull())
    
    return monthly_pizza


def most_common_size_per_month(df_spark):
    # Convert to Pandas DataFrame for plotting
    most_common_sizes_pd = df_spark.toPandas()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=most_common_sizes_pd, 
                x='month', 
                y='quantity',
                hue='pizza_size', 
                palette='viridis',
                width=1.2
                )
    plt.title('Most Common Pizza Size per Month', fontsize=14)
    plt.xlabel('')
    plt.ylabel('Number of Orders', fontsize=14)
    plt.legend(title='Pizza Size', fontsize=14)
    plt.xticks(range(0, 12), 
               ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 
               rotation=45,
               fontsize=14,
               ha='right')
    plt.grid(True)
    plt.show()

def main():
    df_spark = load_data('../pizza_sales.csv')
    df_monthly = convert_to_datetime(df_spark)
    most_common_size_per_month(df_monthly)

if __name__ == '__main__':
    main()