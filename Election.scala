// Databricks notebook source
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.util.IntParam
import org.apache.spark.util.StatCounter
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.rdd._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.clustering.{KMeans,KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import scala.io.Source
import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import java.io.File
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._
import sqlContext._
import org.apache.spark.sql.functions.{col, lit, when}

// COMMAND ----------

val schema1 = StructType(Array(
             StructField("state",StringType, true),
             StructField("state_abbr",StringType, true),
             StructField("county",StringType, true),
             StructField("fips",StringType, true),
             StructField("party",StringType, true),
             StructField("candidate",StringType, true),
             StructField("votes",IntegerType, true),
             StructField("fraction_votes",DoubleType, true)))

// store the dataset in the defined schema
val df = spark.read.option("header","true").schema(schema1).csv("/FileStore/tables/results.csv")

//filter the dataset into democrat and republic
val df_r = df.filter($"party" === "Republican")
val df_d = df.filter($"party" === "Democrat")

//create the view for democrat data
df_d.createOrReplaceTempView("election")


//select the data for only the winning candidate in each county
val election1 = spark.sql("""SELECT * from election INNER JOIN (SELECT fips as b, MAX(fraction_votes) 
AS a FROM election GROUP by fips) 
groupedtt WHERE election.fips = groupedtt.b AND election.fraction_votes = groupedtt.a""")

election1.createOrReplaceTempView("election1")

val d_winner = spark.sql("SELECT state, state_abbr, county, fips, party, candidate, votes, fraction_votes FROM election1")
d_winner.createOrReplaceTempView("democrat")

val d_state = spark.sql("SELECT state, candidate, count(candidate) as countyswon from democrat group by state, candidate")

d_state.createOrReplaceTempView("state")


// COMMAND ----------

val schema2 = StructType(Array(StructField("fips",StringType,true), 
StructField("area_name",StringType, true),
StructField("state_abbreviation",StringType, true),
StructField("Population_2014",IntegerType, true),
StructField("Population_2010_April",IntegerType, true),
StructField("Change_in_Population_Percent",DoubleType, true), 
StructField("Population_2010",IntegerType, true),
StructField("People_under_5",DoubleType, true),
StructField("People_under_18",DoubleType, true),
StructField("People_over_65",DoubleType, true),   
StructField("Female_percentage",DoubleType, true),                              
StructField("White",DoubleType, true),
StructField("Black",DoubleType, true),
StructField("Native",DoubleType, true),
StructField("Asian",DoubleType, true),
StructField("Pacific_Islands",DoubleType, true),
StructField("Two_or_more_race",DoubleType, true),
StructField("Hispancis_or_latino",DoubleType, true),
StructField("Non_hispanic_white",DoubleType, true),                              
StructField("Same_house_1plus_years",DoubleType, true),
StructField("Foreign_born",DoubleType, true),
StructField("Language_other_than_english",DoubleType, true),
StructField("High_School",DoubleType, true),
StructField("Bachelor",DoubleType, true),
StructField("Veteran",DoubleType, true),
StructField("Travel_Time_to_work",DoubleType, true),                                                             
StructField("Housing_unit",IntegerType, true),
StructField("Home_Ownership_Rate",DoubleType, true),
StructField("Housing_unit_in_multi_unit_structure",DoubleType, true),
StructField("Median_value_of_owner_ocuupied_units",IntegerType, true), 
StructField("Households",IntegerType, true),
StructField("Person_per_household",DoubleType, true),
StructField("Per_capita_money_income",IntegerType, true),
StructField("Median_household_income",IntegerType, true),
StructField("Persomn_below_poverty_line",DoubleType, true),                               
StructField("Private_nonfarm_establishment",IntegerType, true),
StructField("Private_nonfarm_employment",IntegerType, true),
StructField("Private_non_farm_employment_percent_change",IntegerType, true),
StructField("Non_employer_establishment",IntegerType, true),                              
StructField("Total_number_of_firms",IntegerType, true),
StructField("Black_owned_firms",DoubleType, true),                             
StructField("Native_owned_firms",DoubleType, true),
StructField("Asian_owned_firms",DoubleType, true),
StructField("Women_owned_firms",DoubleType, true),                               
StructField("Manufacturer_shipments",DoubleType, true),
StructField("Merchant_wholesale_sale",DoubleType, true),                               
StructField("Retail_sales",DoubleType, true),                               
StructField("Retail_sales_per_capita",IntegerType, true),
StructField("Accomodation_and_food_sales",IntegerType, true),
StructField("Building_permits",IntegerType, true),
StructField("Land_area_in_sq_miles",DoubleType, true),
StructField("Population_per_sq_mile",DoubleType, true)))

// COMMAND ----------

val df2 = sqlContext.read.option("header","true").schema(schema2).csv("/FileStore/tables/county_facts.csv")

df2.createOrReplaceTempView("facts")

val df_facts = spark.sql("SELECT facts.fips as fips, democrat.state as state, facts.state_abbreviation as state_abbreviation, area_name, candidate, People_over_65, Female_percentage, White, Black, Asian, Hispancis_or_latino, Foreign_born, Language_other_than_english, Bachelor, Veteran, Home_Ownership_Rate, Median_household_income, Persomn_below_poverty_line, Population_per_sq_mile from FACTS INNER JOIN democrat on CAST(facts.fips as INT) = CAST(democrat.fips as INT) ")

df_facts.createOrReplaceTempView("winner_facts")

val hc = df_facts.filter($"candidate" === "Hillary Clinton")
val bs = df_facts.filter($"candidate" === "Bernie Sanders")

val whc = hc.withColumn("w_hc", lit(1)).withColumn("w_bs",lit(0))
val wbs = bs.withColumn("w_hc", lit(0)).withColumn("w_bs",lit(1))

whc.createOrReplaceTempView("whc")
wbs.createOrReplaceTempView("wbs")

val result = spark.sql("SELECT * from whc union ALL select * from wbs")
result.createOrReplaceTempView("result")

// COMMAND ----------

// Building the machine learning model

val featureCols = Array("People_over_65", "Female_percentage", "White", "Black", "Asian",
                         "Hispancis_or_latino", "Foreign_born", "Language_other_than_english",
                         "Bachelor", "Veteran",
                         "Home_Ownership_Rate", "Median_household_income", "Persomn_below_poverty_line",
                         "Population_per_sq_mile", "w_hc", "w_bs")

val rows = new VectorAssembler().setInputCols(featureCols).setOutputCol("features").transform(result)
val kmeans = new org.apache.spark.ml.clustering.KMeans().setK(4).setFeaturesCol("features").setPredictionCol("prediction")
val model = kmeans.fit(rows)
model.clusterCenters.foreach(println)
val categories = model.transform(rows)
categories.createOrReplaceTempView("model_output")

