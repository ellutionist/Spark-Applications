import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}

object NetworkWordCount {
  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage: NetworkWordCount <application id> <sql-host:sql-port> <repartition_num> <source_hostname1:port1, ...>")
      System.exit(1)
    }

    val appId = args(0).toInt

    val sqlHostAndPort = args(1).split(":")
    val sqlHost = sqlHostAndPort(0)
    val sqlPort = sqlHostAndPort(1).toInt

    val repartitionNum = args(2).toInt

    StreamingExamples.setStreamingLogLevels()

    val sparkConf = new SparkConf().setMaster("local").setAppName("test")
    val ssc = new StreamingContext(sparkConf, Seconds(2))

    ssc.addStreamingListener(new DelayListener(appId, sqlHost, sqlPort, ssc))

    val receiverArgs = args(3).split(",")
    val numStreams: Int = receiverArgs.length

    val lines = (1 to numStreams).map { i =>
      val sourceHostAndPort = receiverArgs(i - 1).split(":")
      val sourceHost = sourceHostAndPort(0)
      val sourcePort = sourceHostAndPort(1).toInt
      ssc.socketTextStream(sourceHost, sourcePort, StorageLevel.MEMORY_ONLY)
    }

    val unifiedStream = ssc.union(lines)

    /* 
    val unifiedStream2 = {
      if (repartitionNum > 0) {
        unifiedStream.repartition(repartitionNum)
      } else {
        unifiedStream
      }
    }

     */

    val words = unifiedStream.flatMap(_.split(" "))
    val wordCounts = words.map(x => (x, 1)).reduceByKey(_ + _)

    /*
    wordCounts.foreachRDD { rdd =>
      if (!rdd.isEmpty()) {
        emptyCount = 0
      } else {
        emptyCount += 1
        println(s"$emptyCount empty.")
        if (emptyCount == 5) {
          ssc.stop(false, false)
          System.exit(1)
        }
      }
    }*/
    wordCounts.print()
    ssc.start()
    ssc.awaitTermination()
  }
}