name := "tfidflda"

version := "0.1"

scalaVersion := "2.11.12"

// https://mvnrepository.com/artifact/org.scalatest/scalatest
libraryDependencies += "org.scalatest" %% "scalatest" % "3.1.0" % Test

libraryDependencies+=  "org.apache.spark" %% "spark-core" % "2.4.4" % "provided"

libraryDependencies+=  "org.apache.spark" %% "spark-sql" % "2.4.4" % "provided"
libraryDependencies  += "org.apache.spark" %% "spark-mllib" % "2.4.4" //% "runtime"

assemblyJarName in assembly := "main.jar"
mainClass in assembly := Some("de.htw.ai.progkonz.SparkApp")

assemblyMergeStrategy in assembly := {
  //discard all files in meta-inf directories
  case PathList("META-INF", xs@_*) => MergeStrategy.discard
  //else if files have the same name (e.g application.config.json) use the first one found in the class path tree
  case x => MergeStrategy.first
}