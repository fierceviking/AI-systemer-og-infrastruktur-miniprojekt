from pie_chart import load_data
from pyspark.sql.functions import col, sum, split, explode
import matplotlib.pyplot as plt


def bar_chart(df, filter_size, feature='pizza_ingredients'):
    # Split the specified column
    df_split = df.withColumn(feature, split(col(feature), ', '))
    df_split.select('pizza_ingredients').show()

    # Make each ingredient have its own row
    df_exploded = df_split.withColumn(feature, explode(col(feature)))   

    # GroupBy the ingredients and quantity
    df_feature = df_exploded.groupBy(feature).agg(sum('quantity').alias('quantity'))

    # Convert to pandas DF
    df_pd = df_feature.toPandas()

    # Filter the dataframe
    df_filtered = df_pd[df_pd['quantity']>filter_size]

    plt.figure(figsize=(12,8)).set_figheight(6)
    plt.bar(
        x=df_filtered.loc[:,feature], 
        height=df_filtered.loc[:,'quantity'],
        label=df_filtered[feature],
        width=.6
        )
    plt.title(f"Bar chart of {feature} (>{filter_size} quantities)")
    plt.xticks(rotation=90, ha='right',fontsize=10)
    plt.ylabel("Quantity")
    plt.tight_layout()
    plt.show()

def main():
    df = load_data('../pizza_sales.csv')
 
    bar_chart(df, filter_size=0)

if __name__ == '__main__':
    main()