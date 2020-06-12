import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object KMeansApplication {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val inputFile = args(0);
    val k = Integer.valueOf(args(1))
    val maxIter = Integer.valueOf(args(2))
    var repartitionCount = 0
    if (args.length >= 4) {
      repartitionCount = args(3).toInt
    }
    var minPartitions = 1
    if (args.length >= 5) {
      minPartitions = Math.max(args(4).toInt, 1)
    }

    val conf = new SparkConf()
    val sc = new SparkContext(conf)
    val textFile = sc.textFile(inputFile, minPartitions = minPartitions)
    val dataset = textFile.map(s => Vectors.dense(s.split(" ").filter(_ != "").map(_.toDouble)))

    val model: KMeansModel = {
      if (repartitionCount>0) {
        val dataset2 = dataset.repartition(repartitionCount)
        dataset2.cache()

        /*
        val partitionCount = dataset2.mapPartitionsWithIndex{
          (partIdx,iter) => {
            var part_map = scala.collection.mutable.Map[String,Int]()
            while(iter.hasNext){
              var part_name = "part_" + partIdx;
              if(part_map.contains(part_name)) {
                var ele_cnt = part_map(part_name)
                part_map(part_name) = ele_cnt + 1
              } else {
                part_map(part_name) = 1
              }
              iter.next()
            }
            part_map.iterator
          }
        }.collect
        partitionCount.foreach(println)
        */

        KMeans.train(dataset2, k, maxIter)
      } else {
        dataset.cache()
        KMeans.train(dataset, k, maxIter)
      }
    }

    println("Cluster centres:")
    for(c <- model.clusterCenters) {
      println(" " + c.toString)
    }
  }
}
