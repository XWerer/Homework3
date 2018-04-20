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
        //SparkConf conf = new SparkConf(true).setAppName("Homework 3");
        //JavaSparkContext sc = new JavaSparkContext(conf);

        ArrayList<Vector> points = InputOutput.readVectorsSeq(args[0]);

        //time for k-center algorithm
        long start = System.currentTimeMillis();
        ArrayList<Vector> centers0 = kcenter(points,10);
        long end = System.currentTimeMillis();
        System.out.println("Time for k-center algorithm is: " + (end - start) + " ms");

        //For viewing the centers computed by k-center
        System.out.println("The number of centers of k-center algorithm are: " + centers0.size() + " and are:");
        for(Vector center : centers0)
            System.out.println(center);

        //We need to clean and reload the points from the file because we modify they by using k-means
        points.clear();
        points = InputOutput.readVectorsSeq(args[0]);

        //Initialization of the weights
        ArrayList<Long> weights = new ArrayList<>();
        for (int i = 0; i < points.size(); i++)
            weights.add(1L);

        //time for k-means++ algorithm
        start = System.currentTimeMillis();
        ArrayList<Vector> centers1 = kmeansPP(points, weights,10);
        end = System.currentTimeMillis();
        System.out.println("Time for k-means++ algorithm is: " + (end - start) + " ms");

        //For viewing the centers computed by k-means++
        System.out.println("The number of centers of k-means++ algorithm are: " + centers1.size() + " and are:");
        for(Vector center : centers1)
            System.out.println(center);
    }

    //Method k-center with time complexity O( |P| * k )
    private static ArrayList<Vector> kcenter (ArrayList<Vector> P ,int k){
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
                        indice = j;
                    }
                }
                else{//negli altri
                    temp = dist.get(j) + Vectors.sqdist(centers.get(i), P.get(j));
                    dist.set(j,temp);
                    if(dist.get(j)> distanza) { //trovo il punto più distante da il resto dei punti
                        max = P.get(j);
                        distanza = dist.get(j);
                        indice=j;
                    }
                }
            }//fine for P
            primo = false;
            centers.add(max);           //quando lho trovato lo aggiungo a quelli trovati e lo
            P.remove(max);              //tolgo così non lo conisidero più
            //P.remove(indice);         //forse meglio in termini di efficenza questo
            dist.remove(indice);        //tolgo anche la distanza annessa
        }//fine for k
        return centers;
    }

    //Method k-means++ weighed with time complexity O( |P| * k )
    private static ArrayList<Vector> kmeansPP(ArrayList<Vector> points, ArrayList<Long> weights, int k){
        ArrayList<Vector> centers = new ArrayList<>(); //ArrayList for the centers
        ArrayList<Double> dist = new ArrayList<>(); //ArrayList for the distances
        //Now we pick the first center randomly and remove it from the points and the weights
        int firstCenter = (int)(Math.random()*points.size());
        centers.add(points.remove(firstCenter));
        weights.remove(firstCenter);
        double distance, totalDistance = 0;
        boolean firstIt = true; //For manage the first center
        for (int i = 0; i < k-1; i++){
            for (int j = 0; j < points.size(); j++){
                if(firstIt){
                    //In the first iteration we need to compute only the distance
                    distance = Vectors.sqdist(points.get(j), centers.get(0));
                    dist.add(distance);
                    totalDistance += distance;
                }
                else{
                    //In a generic iteration we need to take the minimum distance
                    distance = Vectors.sqdist(points.get(j), centers.get(i-1));
                    if(distance < dist.get(j)){
                        dist.set(j, distance);
                        totalDistance += distance;
                    }
                    else
                        totalDistance += dist.get(j);
                }
            }
            //We go to choose  the next center with the probability distribution
            double r = Math.random(); //For a random value
            double value = 0; //The incrementation of the probability
            for(int j = 0; j < points.size(); j++){
                value += dist.get(j) * weights.get(j) / totalDistance;
                if (r < (value)) {
                    //In this case we can take the new center and remove the corresponding value from the other List
                    centers.add(points.remove(j-1));
                    dist.remove(j-1);
                    weights.remove(j-1);
                    break;
                }
            }
            totalDistance = 0;
            firstIt = false;
        }
        return centers;
    }
}
