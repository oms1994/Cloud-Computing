import java.io.*;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;


class Elem implements Writable {
    public int tag;  // 0 for M, 1 for N
    public int index;  // one of the indexes (the other is used as a key)
    public double value;


    Elem() {
    }

    Elem(int t, int ind, double val) {
        tag = t;
        index = ind;
        value = val;
    }

    public void write(DataOutput out) throws IOException {
        out.writeInt(tag);
        out.writeInt(index);
        out.writeDouble(value);
    }

    public void readFields(DataInput in) throws IOException {
        tag = in.readInt();
        index = in.readInt();
        value = in.readDouble();
    }

}

class Pair implements WritableComparable<Pair> {
    public int i;
    public int j;

    Pair() {}

    Pair(int row, int col) {
        this.i = row;
        this.j = col;
    }

    public void write(DataOutput out) throws IOException {
        out.writeInt(this.i);
        out.writeInt(this.j);
    }

    public void readFields(DataInput in) throws IOException {
        this.i = in.readInt();
        this.j = in.readInt();
    }


    //comparable
    public int compareTo(Pair p) {

        if (this.i == p.i)

        {

            if (this.j == p.j)

                return 0;

            else if (this.j < p.j)

                return -1;

            else

                return 1;

        }
        if (this.i < p.i) {

            return -1;
        }
        else

            return 1;

    }


    public String toString() {
        return this.i + "\t" + this.j;
    }


}

public class Multiply {
    //MATRIX M
    public static class Mapper1 extends Mapper<Object, Text, IntWritable, Elem> {
        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            Scanner s = new Scanner(value.toString()).useDelimiter(",");

            int i = s.nextInt();
            int j = s.nextInt();
            double v = s.nextDouble();
            int t = 0;
            System.out.println("I VAL" + i);
            System.out.println("J VAL" + j);
            System.out.println("DATA" + v);
            context.write(new IntWritable(j), new Elem(t, i, v));
            s.close();
        }
    }


    //MATRIX N
    public static class Mapper2 extends Mapper<Object, Text, IntWritable, Elem> {
        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            Scanner s = new Scanner(value.toString()).useDelimiter(",");

            int i = s.nextInt();
            int j = s.nextInt();
            double v = s.nextDouble();
            int t = 1;
            System.out.println("I VAL" + i);
            System.out.println("J VAL" + j);
            System.out.println("DATA" + v);
            context.write(new IntWritable(i), new Elem(t, j, v));
            s.close();
        }
    }

    public static class Reducer1 extends Reducer<IntWritable, Elem, Pair, DoubleWritable> {

        static Vector<Elem> M_MatrixVec = new Vector<Elem>();
        static Vector<Elem> N_MatrixVec = new Vector<Elem>();


        public void reduce(IntWritable key, Iterable<Elem> values, Context context)

                throws IOException, InterruptedException {

            M_MatrixVec.clear();

            N_MatrixVec.clear();

            for (Elem e : values) {

                System.out.println("tag "+e.tag+"index "+e.index+"value "+e.value);
                if (e.tag == 0) {

                    Elem m = new Elem(e.tag, e.index, e.value);

                    M_MatrixVec.add(m);

                } else {

                    Elem n = new Elem(e.tag, e.index, e.value);

                    N_MatrixVec.add(n);

                }

            }

            for (Elem a : M_MatrixVec) {

                for (Elem b : N_MatrixVec) {
                    context.write(new Pair(a.index, b.index), new DoubleWritable(a.value * b.value));

                    System.out.println("aindex "+a.index+"bindex "+b.index+"avalue*bval "+a.value*b.value);

                }

            }
        }


    }

    public static class Mapper3 extends Mapper<Pair, DoubleWritable, Pair, DoubleWritable> {
        @Override
        public void map(Pair key, DoubleWritable value, Context context)
                throws IOException, InterruptedException {
            context.write(key, value);
            System.out.println("Key : "+key+"value :"+value);
        }
    }

    public static class FinalReducer extends Reducer<Pair, DoubleWritable, Pair, DoubleWritable> {


        public void reduce(Pair key, Iterable<DoubleWritable> values, Context context)

                throws IOException, InterruptedException {


            double final_value = 0;

            for (DoubleWritable v : values) {

                final_value = final_value + v.get();
                System.out.println("Filnal Val" + final_value+ "Val"+v.get());

            }


            context.write(key, new DoubleWritable(final_value));

        }

    }


    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job1 = Job.getInstance(conf);
        job1.setJobName("MatMul1");
        job1.setJarByClass(Multiply.class);
        job1.setOutputKeyClass(Pair.class);
        job1.setOutputValueClass(DoubleWritable.class);
        job1.setMapOutputKeyClass(IntWritable.class);
        job1.setMapOutputValueClass(Elem.class);
        job1.setReducerClass(Reducer1.class);
        //job1.setInputFormatClass(TextInputFormat.class );
        job1.setOutputFormatClass(SequenceFileOutputFormat.class);
        // job1.setMapperClass(Mapper1.class);
        MultipleInputs.addInputPath(job1, new Path(args[0]), TextInputFormat.class, Mapper1.class);
        MultipleInputs.addInputPath(job1, new Path(args[1]), TextInputFormat.class, Mapper2.class);
        FileOutputFormat.setOutputPath(job1, new Path(args[2]));
        job1.waitForCompletion(true);
        Configuration conf2 = new Configuration();
        Job job2 = Job.getInstance(conf2);
        job2.setJobName("MatMul2");
        job2.setJarByClass(Multiply.class);
        job2.setOutputKeyClass(Pair.class);
        job2.setOutputValueClass(DoubleWritable.class);
        job2.setMapOutputKeyClass(Pair.class);
        job2.setMapOutputValueClass(DoubleWritable.class);
        job2.setMapperClass(Mapper3.class);
        job2.setReducerClass(FinalReducer.class);
        job2.setInputFormatClass(SequenceFileInputFormat.class);
        job2.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.setInputPaths(job2, new Path(args[2]));
        FileOutputFormat.setOutputPath(job2, new Path(args[3]));
        job2.waitForCompletion(true);
    }
}