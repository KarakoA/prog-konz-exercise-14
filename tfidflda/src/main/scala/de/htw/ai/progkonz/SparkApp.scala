package de.htw.ai.progkonz

import org.apache.log4j.{BasicConfigurator, Level, Logger}
import org.apache.spark.ml.clustering.{LDA, LDAModel}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, Tokenizer}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.reflect.io.Path

//copied from the notebook and midly adapted
//first arg - path to /data/en/ folder
object SparkApp {

  def main(args: Array[String]): Unit = {
    BasicConfigurator.configure()
    Logger.getRootLogger.setLevel(Level.ERROR)

    val sparkSession = SparkSession.builder().master("local[*]").appName("ProgKonz").getOrCreate()

    println(f"Data Path - ${args(0)}")
    val (countsDF, vocabulary) = countsF(args(0))
    val (tfidfDF, _) = tfIdf(countsDF, vocabulary)

    //train
    val (ldaModel, _) = fitLDAModel(tfidfDF, vocabulary)

    // compute
    val ldaTopicsDF = ldaModel.transform(tfidfDF).collect()

    println(f"Done")

    //close session
    sparkSession.close()
  }

  def load(path: String): DataFrame = {
    val sc = SparkSession.builder().getOrCreate()
    import sc.implicits._
    //loads the files and returns them as (title, content) pairs
    val dataRDD =sc.sparkContext.wholeTextFiles(path)
      //map the paths to only the file names
      .map { case (fileName, content) => (Path(fileName).name.replace(".txt", ""), content) }

    //convert them to a dataframe, naming the title column to 'label' and the content to 'text'
    val dataDF = dataRDD.toDF("label", "text")
    dataDF.repartition(16)
  }

  val loadF: String => DataFrame = load _
  val countsF: String => (DataFrame, Array[String]) = loadF andThen tokenize andThen preprocess andThen counts

  def tokenize(inputDF: DataFrame): DataFrame = {
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    //tokenize using a standard whitespace tokenizer
    val tokenizedDF = tokenizer.transform(inputDF)
    tokenizedDF
  }

  def preprocess(inputDF: DataFrame): DataFrame = {
    val sc = SparkSession.builder().getOrCreate()
    import sc.implicits._

    val removeNonAlphanumeric: Seq[String] => Seq[String] = _.map(
      word => word
        .replaceAll("[^A-z0-9]", ""))
      .filter(_.nonEmpty)
    val removeNonAlphanumericUDF = udf(removeNonAlphanumeric)
    val preprocessedDF = inputDF.withColumn("words", removeNonAlphanumericUDF('words))
    preprocessedDF
  }

  def counts(inputDF: DataFrame): (DataFrame, Array[String]) = {
    // get a CountVectorizerModel
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      //used atleast minTF times, to prevent spelling mistakes
      .setMinTF(3)
      .fit(inputDF)
    val vocabulary = cvModel.vocabulary
    val countsDF = cvModel.transform(inputDF)
    (countsDF, vocabulary)
  }

  def tfIdf(inputDF: DataFrame, vocabulary: Array[String]): (DataFrame, Array[String]) = {
    //use the already computed counts to build a TF-IDF model
    val tfidfModel = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .fit(inputDF)
    val TFIDFDF = tfidfModel.transform(inputDF)
    (TFIDFDF, vocabulary)
  }


  def fitLDAModel(inputDF: DataFrame, vocabulary: Array[String]): (LDAModel, Array[String]) = {
    val cachedInput = inputDF.cache
    // Trains a LDA model.
    val ldaModel = new LDA()
      .setK(10)
      .setMaxIter(25)
      .fit(cachedInput)
    (ldaModel, vocabulary)
  }

}
