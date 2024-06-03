from pyspark.sql import SparkSession

spark = SparkSession.builder\
    .appName("QuickSort")\
    .getOrCreate()

sc = spark.sparkContext

def quickSort(list):
    print(list)
    
    if len(list) <= 1:
        return list
    else:
        pivot = list[0]
        less = [x for x in list[1:] if x <= pivot]
        greater = [x for x in list[1:] if x > pivot]
        return quickSort(less) + [pivot] + quickSort(greater)

df = spark.read.option("header", True).csv("Data.csv")
#df = spark.read.csv('/user/angeles.marinbatana/Data.csv', header=True, inferSchema=True)
rdd = sc.parallelize(df.collect())
result = rdd.mapPartitions(lambda x: [quickSort(list(x))]).collect()

print(result)

spark.stop()
