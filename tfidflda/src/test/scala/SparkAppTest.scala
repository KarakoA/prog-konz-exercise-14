import de.htw.ai.progkonz.SparkApp._
import org.apache.log4j.{BasicConfigurator, Level, Logger}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.{Row, SparkSession}
import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalactic.Tolerance._

class SparkAppTest extends AnyFlatSpec with Matchers {
  private val sparkSession: SparkSession = SparkSession.builder().master("local").appName("ProgKonz Test").getOrCreate()

  BasicConfigurator.configure()
  Logger.getRootLogger.setLevel(Level.ERROR)

  implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(0.1)
  val (countsDF, vocabulary) = countsF("data/test")
  val (tfidfDF, _) = tfIdf(countsDF, vocabulary)

  //train
  val (ldaModel, _) = fitLDAModel(tfidfDF, vocabulary)

  // compute
  val ldaTopicsDF: Array[Row] = ldaModel.transform(tfidfDF).collect()


  "vocabulary" should "contain all words" in {
    vocabulary.sorted shouldBe Array("a", "another", "b", "document", "in", "other", "the", "this", "yet")
  }
  "counts" should "contain the correct values" in {
    countsDF.collect().map(x => x.getAs[SparseVector]("rawFeatures")).toSet shouldBe
      Set(
        new SparseVector(9, Array(0, 1, 2, 3, 6, 8), Array(3.0, 3.0, 3.0, 3.0, 3.0, 3.0)),
        new SparseVector(9, Array(0, 1, 2, 7), Array(3.0, 3.0, 3.0, 3.0)),
        new SparseVector(9, Array(0, 1, 3, 4, 5), Array(3.0, 3.0, 3.0, 3.0, 3.0))
      )
  }
  "tfidf" should "contain the correct values" in {
    ldaTopicsDF.map(x => x.getAs[SparseVector]("features")).toSet shouldBe
      Set(
        new SparseVector(9, Array(0, 1, 2, 3, 6, 8), Array(0.0, 0.0, 0.8630462173553426, 0.8630462173553426, 2.0794415416798357, 2.0794415416798357)),
        new SparseVector(9, Array(0, 1, 2, 7), Array(0.0, 0.0, 0.8630462173553426, 2.0794415416798357)),
        new SparseVector(9, Array(0, 1, 3, 4, 5), Array(0.0, 0.0, 0.8630462173553426, 2.0794415416798357, 2.0794415416798357))
      )
  }

  "ldaTopicsDF" should "contain the correct values" in {
    ldaTopicsDF.flatMap(row => row.getAs[DenseVector]("topicDistribution").toArray).sorted
      .zip(Array(0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.015, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.026, 0.772, 0.852, 0.869))
      .foreach{case(e,a)=> e === a +- 0.1}

  }
}
