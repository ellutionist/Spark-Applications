name := "SparkStreaming2"

version := "0.1"

scalaVersion := "2.11.10"

libraryDependencies += "org.postgresql" % "postgresql" % "42.2.12"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.5"

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.4.5"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.5"
