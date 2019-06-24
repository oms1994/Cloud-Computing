import java.io.*;
import java.util.Scanner;
import java.util.*;
import java.lang.Math.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.*;
import java.util.Vector;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;

/**************Classes - Vertex***********/
class Vertex implements Writable {
    public int tag;                 // 0 for a graph vertex, 1 for a group number
    public long group;                // the group where this vertex belongs to
    public long VID=0;                  // the vertex ID
    public Vector<Long>adjacent = new Vector<Long>();     // the vertex neighbors
    /* ... */

    Vertex()
    {}
    public Vertex(int t,long grp,long Ver,Vector<Long>adjt)
    {

        this.tag=t;
        this.group=grp;
        this.VID=Ver;
        this.adjacent=adjt;

    }


    public  Vertex(int t , long grp)
    {
        super();
        this.tag=t;
        this.group=grp;

    }

    public void write(DataOutput out) throws IOException {
        out.writeInt(tag);
        out.writeLong(group);
        out.writeLong(VID);
        int adsize=0;
        adsize=this.adjacent.size();
        out.writeInt(adsize);
        for(int i=0;i<adsize;i++)
        {

            out.writeLong(adjacent.get(i));
        }




    }
    public void readFields(DataInput in) throws IOException {

        Vector<Long> vertexttemp = new Vector<Long>();
        tag = in.readInt();
        group = in.readLong();
        VID = in.readLong();
        int adjsize=0 ;
        adjsize=in.readInt();

        for(long i =0;i<adjsize;i++)
        {
            vertexttemp.add(in.readLong());

        }

        this.adjacent=vertexttemp;

    }


}

/**************Classes - Graph***********/
public class Graph {


    public static class Mapper1 extends Mapper<Object, Text, LongWritable, Vertex> {
        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            Scanner s = new Scanner(value.toString()).useDelimiter(",");
            Vector<Long> VerVec = new Vector<Long>();
            int t = 0;

            long vid = s.nextLong();

            while(s.hasNext())
            {

                VerVec.add(s.nextLong());

            }


            System.out.println("vid VAL" + vid);
            Vertex sendVertex= new Vertex (t, vid, vid, VerVec);

            context.write(new LongWritable(vid),sendVertex );
            s.close();
        }
    }

    public static class Mapper2 extends Mapper<LongWritable, Vertex,LongWritable, Vertex> {
        public void map(LongWritable key, Vertex values, Context context)
                throws IOException, InterruptedException {
            short t=1;
            Vertex sendVertex = new Vertex(t,values.group);
            context.write(new LongWritable(values.VID) ,values);

            for (int i =0;i<values.adjacent.size();i++) {
                context.write(new LongWritable(values.adjacent.get(i)),sendVertex);
            }

        }



    }



    public static class Reducer2 extends Reducer<LongWritable, Vertex, LongWritable, Vertex> {


        public void reduce(LongWritable key, Iterable<Vertex> values, Context context)

                throws IOException, InterruptedException {


            int t = 0;
            Vector<Long> vecclone = new Vector<Long>();
            long m = Long.MAX_VALUE ;
            for (Vertex vcount : values) {
                if (vcount.tag == 0) {
                    vecclone = (Vector) vcount.adjacent.clone();    // found the vertex with vid
                    // m = Math.min(m,(Long)v.group);
                }
                m=Math.min(m,vcount.group);
            }vecclone.toString();

            // long minobj = Long.parseLong(String.valueOf(m));
            //long keyval = Long.parseLong(String.valueOf(key));

            context.write(new LongWritable(m), new Vertex(t,m,key.get(),vecclone));



        }
    }

    public static class FinalMapper extends Mapper<LongWritable,Vertex,LongWritable,LongWritable> {
        public void map(LongWritable key, Vertex values, Context context)
                throws IOException, InterruptedException {
            long t=1;

            context.write(key ,new LongWritable (t));


        }



    }

    public static class FinalReducer extends Reducer<LongWritable, LongWritable, LongWritable, LongWritable> {


        public void reduce(LongWritable group,Iterable <LongWritable> values, Context context)

                throws IOException, InterruptedException {


            long m=0;
            for (LongWritable v : values) {

                m=(m+v.get());



            }
            context.write(group,new LongWritable(m));
        }
    }


    /**************Main ***********/

    public static void main ( String[] args ) throws Exception {
        Configuration con1=new Configuration();
        Job job = Job.getInstance();
        job.setJobName("MyJob");
        job.setJarByClass(Graph.class);
        job.setMapperClass(Mapper1.class);
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(Vertex.class);
        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(Vertex.class);
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);

        // job.setNumReduceTasks(0);
        FileInputFormat.setInputPaths(job,new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]+"/f0"));
        job.waitForCompletion(true);
        for ( short i = 0; i < 5; i++ ) {
            Job job1 = Job.getInstance();
            /* ... Second Map-Reduce job to propagate the group number */
            job1.setJarByClass(Graph.class);
            job1.setJobName("MyJob1");
            job1.setOutputKeyClass(LongWritable.class);
            job1.setOutputValueClass(Vertex.class);
            job1.setMapOutputKeyClass(LongWritable.class);
            job1.setMapOutputValueClass(Vertex.class);
            job1.setInputFormatClass(SequenceFileInputFormat.class);
            job1.setOutputFormatClass(SequenceFileOutputFormat.class);
            job1.setMapperClass(Mapper2.class);
            job1.setReducerClass(Reducer2.class);
            FileInputFormat.setInputPaths(job1,new Path(args[1]+"/f"+i));
            FileOutputFormat.setOutputPath(job1, new Path(args[1]+"/f"+(i+1)));
            job1.waitForCompletion(true);
        }
        Job job2 = Job.getInstance();
        job2.setJobName("MyJob2");
        job2.setJarByClass(Graph.class);
        job2.setOutputKeyClass(LongWritable.class);
        job2.setOutputValueClass(LongWritable.class);
        job2.setMapOutputKeyClass(LongWritable.class);
        job2.setMapOutputValueClass(LongWritable.class);
        job2.setInputFormatClass(SequenceFileInputFormat.class);
        job2.setOutputFormatClass(TextOutputFormat.class);
        job2.setMapperClass(FinalMapper.class);
        job2.setReducerClass(FinalReducer.class);
        FileInputFormat.setInputPaths(job2,new Path(args[1]+"/f5"));
        FileOutputFormat.setOutputPath(job2, new Path(args[2]));
        job2.waitForCompletion(true);


    }
}


