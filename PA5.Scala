import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scala.collection.mutable
import org.apache.spark.rdd.RDD

object Graph {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Join")                           /** Basic Configuration **/
    val sc = new SparkContext(conf)

      var graph = sc.textFile(args(0)).map(line => {
      val input_vertex = line.split(",")                                    /** Reading from File **/
      val adj_group = new Array[Long](input_vertex.length - 1)
      var i=1;

      do {
           adj_group(i-1) = input_vertex(i).toLong
           i=i+1;
          } 
     while(i <=input_vertex.length-1);

 
      (input_vertex(0).toLong, input_vertex(0).toLong, adj_group)        /**Input tuple*/
    })
     
for (j <- 1 to 5)
{

       val groupval = FlatMap_Operation(graph);
       var min_op_red = Find_Min_group(groupval);                       /*Finding Min Group by reduce by key**/      
       var initial_op=graph.map{case(a) => (a._1, a)}
       var Op_joined = min_op_red.join(initial_op)                      /**Join with original graph***/
       val final_graph=create_Final_graph(Op_joined)                    /**reconstruct**/

      graph=final_graph
    
    }

    val map_reduce_final_sort = graph.map(graph => (graph._2, 1)).reduceByKey((x, y) => x + y).sortByKey(true, 0) /*Map amd count*/
    val formated_final_graph =FormatOutput(map_reduce_final_sort) /**Formating  **/
    formated_final_graph.collect().foreach(println)    
    sc.stop()

  }



/***Otput formatting function**/

  def FormatOutput(ResultMatrix:RDD[((Long, Int))]) : RDD[String] = {
    
     val Reuslt_Matrix_formated = ResultMatrix.map { case ((k,v)) =>k+" "+v}  /*Formating Final Matrix*/

     return Reuslt_Matrix_formated

  }

/***Flatmap Operation function***/


 def FlatMap_Operation(ResultVetrex:RDD[(Long, Long, Array[Long])] ): RDD[(Long,Long)] = {
   
        val flatmap_Input = ResultVetrex.flatMap{case(a,b,c) =>
        val grouplen: Int =(c.length) 
        val graphver = new Array[(Long, Long)](grouplen+1)
        graphver(0) = (a, b)
        val adjver: Array[Long] = c
       
        for (index <- 0 to grouplen-1)
         {
          graphver(index + 1) = (adjver(index), b)
         }
           graphver
       
        }

   return flatmap_Input
 }

/***Mingroup function by using Reduce by key to return min group **/
 def Find_Min_group(Mingroup:RDD[(Long,Long)]):RDD[(Long,Long)]={
       val mingrp = Mingroup.reduceByKey((a, b) => { 
      var min_group: Long =0

        if (a <= b) {
          min_group = a
        }
        else {
          min_group = b
        }
        
          min_group
      })

        return mingrp
 }

/***Final graph restructing function***/

def create_Final_graph(reducedgraph:RDD[(Long, (Long, (Long, Long, Array[Long])))] ): RDD[(Long,Long,Array[Long])]={
   
        val graph1=reducedgraph.map{case(a,b) => 
  
        val adj=b._2
        var connected_ver =  (a,b._1,adj._3) /**VID, group, Vertex **/
     
        connected_ver
      }

       return graph1

  
}

}