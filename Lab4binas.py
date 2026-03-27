from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import seaborn as sns

# Create Spark Session
spark = SparkSession.builder.appName("Refined_COVID19_BPS").getOrCreate()

# Load the dataset
df = pd.read_csv(r"C:/spark/spark-3.5.8-bin-hadoop3/bin/World-happiness-report-2024.csv")


# Basic cleaning
df = df.withColumn("Percentage", F.col("Percentage").cast("double"))
df = df.na.drop()

print("Total rows:", df.count())
df.show(5)

# =========================
# 5 Visualizations in PySpark
# =========================

# 1. Average Percentage by Breakdown Category (Firm Size)
avg_by_size = df.groupBy("Breakdown_Category").agg(
    F.avg("Percentage").alias("Average_Percentage")
).orderBy("Average_Percentage", ascending=False)

avg_by_size_pd = avg_by_size.toPandas()

plt.figure(figsize=(10,6))
sns.barplot(x="Average_Percentage", y="Breakdown_Category", data=avg_by_size_pd)
plt.title("Average Percentage by Firm Size (PySpark)")
plt.xlabel("Average Percentage")
plt.ylabel("Firm Size")
plt.tight_layout()
plt.show()

# 2. Top 10 Countries by Average Percentage (National Average only)
top_countries = df.filter(df.Breakdown_Category == "National Average") \
    .groupBy("Country") \
    .agg(F.avg("Percentage").alias("Avg_Percentage")) \
    .orderBy("Avg_Percentage", ascending=False) \
    .limit(10)

top_countries_pd = top_countries.toPandas()

plt.figure(figsize=(10,6))
sns.barplot(x="Avg_Percentage", y="Country", data=top_countries_pd)
plt.title("Top 10 Countries - Average Business Performance (PySpark)")
plt.xlabel("Average Percentage")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# 3. Average Percentage by Year
by_year = df.groupBy("Year").agg(F.avg("Percentage").alias("Average_Percentage"))

by_year_pd = by_year.toPandas()

plt.figure(figsize=(8,5))
sns.barplot(x="Year", y="Average_Percentage", data=by_year_pd)
plt.title("Average Percentage by Year (PySpark)")
plt.show()

# 4. Distribution of Percentage Values (Histogram)
percentages_pd = df.select("Percentage").toPandas()

plt.figure(figsize=(8,5))
plt.hist(percentages_pd["Percentage"], bins=20)
plt.title("Distribution of Percentage Values (PySpark)")
plt.xlabel("Percentage")
plt.ylabel("Frequency")
plt.show()

# 5. Average Percentage for Firm Sizes only (excluding National Average)
firm_sizes = df.filter(df.Breakdown_Type == "Firm Size") \
    .groupBy("Breakdown_Category") \
    .agg(F.avg("Percentage").alias("Avg_Percentage")) \
    .orderBy("Avg_Percentage", ascending=False)

firm_sizes_pd = firm_sizes.toPandas()

plt.figure(figsize=(10,6))
sns.barplot(x="Avg_Percentage", y="Breakdown_Category", data=firm_sizes_pd)
plt.title("Average Performance by Firm Size (PySpark)")
plt.xlabel("Average Percentage")
plt.ylabel("Firm Size Category")
plt.tight_layout()
plt.show()

spark.stop()