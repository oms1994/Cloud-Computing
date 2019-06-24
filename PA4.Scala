import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scala.collection.mutable

object Multiply {
  
  def main(args Array[String]){
   
    val conf  = new SparkConf().setAppName(MatMul)  Setting Configuration 
    val sc = new SparkContext(conf)

    val input1 = sc.textFile(args(0)).map( line = { val a = line.split(,)
                                                (a(0).toInt,a(1).toInt,a(2).toDouble) } ) Reading M from file
    	
    val input2 =sc.textFile(args(1)).map( line = { val a = line.split(,)
                                                (a(0).toInt,a(1).toInt,a(2).toDouble) } ) Reading N from file

	val mapMindex = input1.map( input1 = (input1._2, input1)) Maping J Coulm 
	val mapNindex =	input2.map( input2 = (input2._1,input2))  Maping I Coulm 

	val multiplyJoin = mapMindex .join(mapNindex)   

	val Product_Val = multiplyJoin.map{ case (k, (input1,input2)) = 
										((input1._1,input2._2),(input1._3  input2._3)) } Multiplying Matrix Values


	val reduceroutput = Product_Val.reduceByKey((a,b) = (a+b)).sortByKey(true, 0) Final Matrix 									
	
	val formated_Matrix =FormatOutput(reduceroutput) Formating  

	formated_Matrix.foreach(println)		

	SaveToFile(formated_Matrix,args)		Saving To File Final Matrix

	sc.stop()

}
Function to Save Final Matrix to File
def SaveToFile(FinalOutPutRDD[String],args Array[String]) {
    
      FinalOutPut.saveAsTextFile(args(2))
  }
Function to Format Final Matrix to File
  def FormatOutput(ResultMatrixRDD[((Int, Int), Double)])  RDD[String] = {
    
     val Reuslt_Matrix_formated = ResultMatrix.map { case ((k,v),j) =k+ +v+ +j}  Formating Final Matrix

     return Reuslt_Matrix_formated
  }
}