package it.unipd.dei.bdc1718;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.ArrayList;

public class G08HM3 {

    public static void main(String[] args) throws Exception {
        if (args.length == 0)
            throw new IllegalArgumentException("Expecting the file name on the command line");

        // Setup Spark
        SparkConf conf = new SparkConf(true).setAppName("Homework 3");
        JavaSparkContext sc = new JavaSparkContext(conf);

        ArrayList<Vector> points = InputOutput.readVectorsSeq(args[0]);

        //tempo del k-center
        long start = System.currentTimeMillis();
        ArrayList<Vector> centri = kcenter(points,10);
        long end = System.currentTimeMillis();
        System.out.println("Time for kcenter is: " + (end - start) + " ms");

        //just for debug
        System.out.println("i centri sono: " + centri.size());
        for(int i = 0; i < centri.size(); i++) {
            System.out.println(centri.get(i));
        }
    }

    private static ArrayList<Vector> kcenter (ArrayList<Vector> P ,int k){ //O( |P| * k )
        ArrayList<Vector> centers = new ArrayList<>();
        Vector max = Vectors.zeros(1);  //mi salvo il quello che sta a distanza massima
        ArrayList<Double> dist = new ArrayList<>(); //distanza massima
        double distanza = 0.0;
        double temp = 0.0;
        boolean primo = true;
        //così so quale cancellare dall'array list delle distanze
        int indice = 0;
        //nelle slide c'è scritto prende un punto arbitrario, prendo il primo
        centers.add(P.get(0));
        //lo tolgo così non lo considero nel calcolo delle distanze
        P.remove(0);
        for(int i = 0; i < k - 1; i++ ){//k-1 perchè il primo lo do io
            for(int j = 0; j < P.size(); j++){
                if(primo) {//primo giro devo calcolare la distanza dal primo punto
                    dist.add(Vectors.sqdist(centers.get(0), P.get(j)));
                    if(dist.get(j)> distanza) { //trovo il punto più distante da il resto dal punto inizale
                        max = P.get(j);
                        distanza = dist.get(j);
                        indice=j;
                    }
                }
                else{//negli altri
                    temp=dist.get(j) + Vectors.sqdist(centers.get(i), P.get(j));
                    dist.set(j,temp);
                    if(dist.get(j)> distanza) { //trovo il punto più distante da il resto dei punti
                        max = P.get(j);
                        distanza = dist.get(j);
                        indice=j;
                    }
                }
            }//fine for P
            primo = false;
            centers.add(max);                                        //quando lho trovato lo aggiungo a quelli trovati e lo
            P.remove(max);                                           //tolgo così non lo conisidero più
            dist.remove(indice);                                     //tolgo anche la distanza annessa
        }//fine for k
        return centers;
    }
}
