import java.sql.{Connection, DriverManager}

import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.scheduler.{StreamingListener, StreamingListenerBatchCompleted}
import org.slf4j.LoggerFactory

class DelayListener(val id: Int, val host: String, val port: Int, val ssc: StreamingContext) extends StreamingListener {
  private val logger = LoggerFactory.getLogger("SparkStreamingDelayListener")
  private val sql_id = id
  private val conn_str = s"jdbc:postgresql://${host}:${port}/ellutionist"
  classOf[org.postgresql.Driver]
  private val conn: Connection = DriverManager.getConnection(conn_str, "ellutionist", "p1995620c")
  private val statement = conn.createStatement

  override def onBatchCompleted(batchCompleted: StreamingListenerBatchCompleted): Unit = {
    val batchInfo = batchCompleted.batchInfo
    val numRecords = batchInfo.numRecords
    val totalDelay = batchInfo.totalDelay
    println(numRecords)
    println(s"Total delay: ${totalDelay.get}")

    if (numRecords > 0) {
      statement.execute(String.format(s"INSERT INTO latency VALUES(CURRENT_TIMESTAMP, ${totalDelay.get}, $sql_id, $numRecords);"))
    }
  }
}