import kagglehub
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace
import os
import shutil

"""
To use this script:
1. Download the Kaggle API token from: https://www.kaggle.com/settings
2. Save it as: ~/.kaggle/kaggle.json
3. Make sure you have access to the private dataset on Kaggle
4. Run the script with Spark installed
"""

# Download latest version of the dataset and start Spark
path = kagglehub.dataset_download("ninosabella/private-chicken-dataset")
file_path = os.path.join(path, "chicken_data.csv")

spark = SparkSession.builder.appName("ChickenSpark").getOrCreate()

# Load CSV into Spark DataFrame
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Rename columns
df = df.withColumnRenamed("v1", "chicken1_id") \
       .withColumnRenamed("v2", "chicken2_id") \
       .withColumnRenamed("v3", "contact_duration") \
       .withColumnRenamed("v4", "antenna") \
       .withColumnRenamed("time_", "mean_interaction") \

df.printSchema()

print("Columns renamed successfully")

# Replace commas with dots (for all string columns only)
for col_name, dtype in df.dtypes:
    if dtype == "string":
        df = df.withColumn(col_name, regexp_replace(col(col_name), ",", "."))

# Drop duplicate rows
df = df.dropDuplicates()

# Unify NA labels (replace 'N/A' and 'NaN' with nulls)
df = df.replace(["N/A", "NaN"], None)

# Drop rows with missing values
df = df.dropna()

output_dir = "cleaned_chicken_dataset"
df.coalesce(1).write.mode("overwrite").csv("cleaned_chicken_dataset", header=True)

for filename in os.listdir(output_dir):
    if filename.startswith("part-") and filename.endswith(".csv"):
        src = os.path.join(output_dir, filename)
        dst = os.path.join(output_dir, "chicken_data_cleaned.csv")
        shutil.move(src, dst)


