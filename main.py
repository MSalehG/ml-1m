from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructType, StructField
import sys
import os


if __name__ == '__main__':
    args = sys.argv
    save_dir = args[3]

    #Schema definition for the movie dataset
    movies_schema = StructType([\
        StructField("id", IntegerType(), nullable=False),
        StructField("title", StringType(), nullable=False),
        StructField("genre", StringType(), nullable=False)
                                   ])
    #Schema definition for the rating dataset
    ratings_schema = StructType([\
        StructField("user_id", IntegerType(), nullable=False),
        StructField("movie_id", IntegerType(), nullable=False),
        StructField("rating", IntegerType(), nullable=False),
        StructField("time", StringType(), nullable=True)
        ])

    #Set the spark.sql.files.maxPartitionBytes for the rating dataset to become 4 partitions
    spark = SparkSession.builder\
        .appName("DE_TEST").master("local[*]")\
        .config("spark.sql.shuffle.partitions", "8")\
        .config("spark.sql.files.maxPartitionBytes", "6m")\
        .getOrCreate()


    #Reading the movie dataset with the defined schema
    movies_df = spark.read.option("header", "false") \
        .option("delimiter", "::").schema(movies_schema)\
        .csv(path=args[1])

    #Reading the rating dataset with the defined schema
    ratings_df = spark.read.option("header", "false") \
        .option("delimiter", "::").schema(ratings_schema) \
        .csv(path=args[2])


    #Creating temp views for the dataframees so as to enable spark-sql
    movies_df.createOrReplaceTempView("movies")
    ratings_df.createOrReplaceTempView("ratings")

    #Joining the two datasets with broadcast join
    #since the movie dataset is much smaller
    mr_joined = spark.sql("""SELECT /*+ BROADCASTJOIN (m) */ m.id, m.title, m.genre, r.rating, r.user_id
    FROM movies m join ratings r on m.id = r.movie_id""").cache()

    mr_joined.createOrReplaceTempView("mr_joined")

    #We use SORT BY instead of ORDER BY to avoid needless shuffling and repartitioning into 1 partition

    #spark-sql implementation of Q1 if the question meant windowed min,max,avg
    q1_windowed = spark.sql("""SELECT mr.id, mr.title, mr.genre,
    MIN(mr.rating) OVER (PARTITION BY mr.id) as min,
    MAX(mr.rating) OVER (PARTITION BY mr.id) as max,
    AVG(mr.rating) OVER (PARTITION BY mr.id) as avg
    FROM mr_joined mr
    SORT BY mr.title""")


    #spark-sql implementation of Q1 if the question meant grouped min,max,avg
    q1_grouped = spark.sql("""SELECT mr.id, mr.title, mr.genre,
    MIN(mr.rating) as min, MAX(mr.rating) as max, AVG(mr.rating) as avg 
    FROM mr_joined mr group by mr.id, mr.title, mr.genre
    SORT BY mr.title""")


    #spark-sql implementation of Q2
    q2 = spark.sql("""SELECT * FROM
    (SELECT mr.user_id, mr.title,
     ROW_NUMBER() OVER (PARTITION BY mr.user_id ORDER BY mr.rating DESC) as rk
     FROM mr_joined mr) ranked
     WHERE ranked.rk < 4
     SORT BY ranked.user_id ASC, ranked.rk ASC""")


    q1_grouped.write.parquet(os.path.join(save_dir, "q1_grouped.parquet"))
    q1_windowed.write.parquet(os.path.join(save_dir, "q1_windowed.parquet"))
    q2.write.parquet(os.path.join(save_dir, "q2.parquet"))
