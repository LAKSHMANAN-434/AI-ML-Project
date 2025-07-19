# Databricks notebook source
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("StudentAssignment").getOrCreate()
data = [
    (1, "Alice", "Engineering", 65000),
    (2, "Bob", "Marketing", 58000),
    (3, "Charlie", "Sales", 52000)
]

schema = ["ID", "Name", "Department", "Salary"]
df = spark.createDataFrame(data, schema=schema)
df.show()
df.printSchema()
df.filter(df["Salary"] > 60000).show()
df.groupby("Department").count().show()
df.groupby("Department").avg("Salary").show()

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Spark DataFrames").getOrCreate()
df = spark.read.csv("/Volumes/workspace/default/sam_volume")
df.show()
df.printSchema()

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("EmployeeDataAnalysis") \
    .getOrCreate()

df = spark.read.option("header", True).option("inferSchema", True).csv("/Volumes/workspace/default/sam_volume")
df.show()

df.printSchema()
df.groupBy("JOB_ID").avg("SALARY").show()

df = df.withColumn("Bonus", df.SALARY * 0.10)
df.show()

df.filter(df.SALARY > 70000).show()

df.groupBy("JOB_ID").avg("SALARY").display()

# Convert Spark DataFrame to Pandas DataFrame
pandas_df = df.toPandas()

# Plotting using Matplotlib
plt.figure(figsize=(12, 6))
plt.bar(pandas_df['JOB_ID'], pandas_df['SALARY'])
plt.xlabel('Job ID')
plt.ylabel('Salary')
plt.title('Salary by Job ID')
plt.show()

