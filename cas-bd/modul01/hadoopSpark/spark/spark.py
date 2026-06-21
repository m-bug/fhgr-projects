from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col

# Spark Session starten
spark = SparkSession.builder.appName("WordCount").getOrCreate()

# Daten einlesen, transformieren und zählen (alles im RAM)
results = (spark.read.text("input.txt")
           .select(explode(split(col("value"), " ")).alias("word"))
           .groupBy("word")
           .count())

results.show()
spark.stop()