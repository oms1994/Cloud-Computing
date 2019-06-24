import org.apache.spark.graphx.{Graph=>Graph1, VertexId,Edge}
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx.util.GraphGenerators
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf


object Graph {
  def main ( args: Array[String] ){
    val conf=new SparkConf().setAppName("Pregel Graph")
    val sc = new SparkContext(conf)
    
    val inputval =sc.textFile(args(0)).map(line => {
    val (gvertex,adjgrp)=line.
                              split(",").splitAt(1)
                                         (gvertex(0).toLong,
                                          adjgrp.toList.map(_.toLong))})

    val inputedge=create_Final_graph(inputval)                                                                 
    val newGraph: Graph1[Long,Long]=createGraphfronEdges(inputedge)
    var i=5;
    
    val ConCpnts=newGraph.pregel(Long.MaxValue,i)(
        (id,Prevgroup,newGroup)=> math.min(Prevgroup,newGroup),
        triplet=>{
            if(triplet.attr<triplet.dstAttr){
              Iterator((triplet.dstId,triplet.attr))
            }else if((triplet.srcAttr<triplet.attr)){
              Iterator((triplet.dstId,triplet.srcAttr))
            }else{
              Iterator.empty
            }
        },
        (c1,c2)=>math.min(c1,c2))

       val res = ConCpnts.vertices

       val countvertex= res.map(newGraph=>(newGraph._2,1))

       val  reducedvertices=countvertex.reduceByKey(_ + _).sortByKey()

       Printoutput(reducedvertices)
      
  }


def create_Final_graph(reducedgraph:RDD[(Long,List[Long])]):RDD[Edge[Long]]={
   
        val inputgraph=reducedgraph.flatMap(a=> a._2.map(b=>(a._1,b)))
                                        
       val createdEdgeRDd=inputgraph.map(gnode=>Edge(gnode._1,gnode._2,gnode._1))
  

        return createdEdgeRDd
      }

def createGraphfronEdges(Edges:RDD[Edge[Long]]):Graph1[VertexId,Long]={   

  val interGraph= Graph1.fromEdges(Edges,"defaultProperty").mapVertices((id,_)=>id)

    return interGraph
}

def Printoutput(preggrap:RDD[(Long, Int)] )
{
       val finalgraph=preggrap.map(keycom=>keycom._1.toString+" "+keycom._2.toString )

          finalgraph.collect().foreach(println)


}


}

 