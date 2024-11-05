import spark

def load_data(file_name):
    # Define data path
    data = os.path.join(os.path.dirname(__file__), file_name)

    # Read data
    df_spark = spark.read.csv(data, header=True, inferSchema=True)
    return df_spark



def main():
    df = load_data('pizza_sales.csv')

if __name__ == '__main__':
    main()