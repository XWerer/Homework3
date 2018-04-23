package it.unipd.dei.bdc1718;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import java.util.ArrayList;
import java.util.Collections;

public class G08HM3 {

    public static void main(String[] args) throws Exception {
        if (args.length != 3)
            throw new IllegalArgumentException("Expecting 3 parameters (file name, k, k1)");

        //Some variables
        long start, end; //Variables for measuring the time of the algorithm
        int k, k1;
        ArrayList<Vector> centers;

        //Take the value k and k1 form the command line
        try {
            k = Integer.valueOf(args[1]);
            k1 = Integer.valueOf(args[2]);
        }
        catch (Exception e) {
            throw new IllegalArgumentException("k and k1 must be integer");
        }

        if (k >= k1)
            throw new IllegalArgumentException("Expecting k < k1");

        //Read the set P from the file passed in input
        ArrayList<Vector> points = InputOutput.readVectorsSeq(args[0]);

        //Run k-center algorithm and measure the time
        start = System.currentTimeMillis();
        centers = kcenter(points, k);
        end = System.currentTimeMillis();
        System.out.println("\nTime for k-center algorithm is: " + (end - start) + " ms");

        //For viewing the centers computed by k-center
        System.out.println("The number of centers of k-center algorithm are: " + centers.size() + " and are:");
        for(Vector center : centers)
            System.out.println(center);

        //We need to add the centers to the set P because the algorithm above modify it
        points.addAll(centers);

        //Initialization of the weights
        ArrayList<Long> weights = new ArrayList<>();
        for (int i = 0; i < points.size(); i++)
            weights.add(1L);

        //Run k-means++ algorithm and measure the time
        start = System.currentTimeMillis();
        centers = kmeansPP(points, weights, k);
        end = System.currentTimeMillis();
        System.out.println("\nTime for k-means++ algorithm is: " + (end - start) + " ms");

        //For viewing the centers computed by k-means++
        System.out.println("The number of centers of k-means++ algorithm are: " + centers.size() + " and are:");
        for(Vector center : centers)
            System.out.println(center);

        //Now we can run the k-meansObj algorithm
        System.out.println("\nThe average of the distances is " + kmeansObj(points, centers));

        //We need to add the centers to the set P because the algorithm above modify it
        points.addAll(centers);

        //Now we are going to extract a coreset where we can run k-means++
        //The first thing to do is to run k-center
        centers = kcenter(points, k1);

        //Now we need to create the weights for centers
        weights.clear();
        for (int i = 0; i < k1; i++)
            weights.add(1L);

        //Now we can run k-means++ for extract the coreset from centers
        ArrayList<Vector> coreSet = kmeansPP(centers, weights, k);

        //Now print the returned value from k-meansObj
        System.out.println("\nThe average of the distances in the coreset is: " + kmeansObj(points, coreSet));
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
    private static ArrayList<Vector> kmeansPP(ArrayList<Vector> P, ArrayList<Long> weights, int k){
        ArrayList<Vector> centers = new ArrayList<>(); //ArrayList for the centers
        ArrayList<Double> dist = new ArrayList<>(); //ArrayList for the distances
        //Now we pick the first center randomly and remove it from the points and the weights
        int firstCenter = (int)(Math.random()*P.size());
        centers.add(P.remove(firstCenter));
        weights.remove(firstCenter);
        double distance, totalDistance = 0;
        boolean firstIt = true; //For manage the first center
        for (int i = 0; i < k-1; i++){
            for (int j = 0; j < P.size(); j++){
                if(firstIt){
                    //In the first iteration we need to compute only the distance
                    distance = Vectors.sqdist(P.get(j), centers.get(0));
                    dist.add(distance);
                    totalDistance += distance;
                }
                else{
                    //In a generic iteration we need to take the minimum distance
                    distance = Vectors.sqdist(P.get(j), centers.get(i-1));
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
            for(int j = 0; j < P.size(); j++){
                value += dist.get(j) * weights.get(j) / totalDistance;
                if (r < (value)) {
                    //In this case we can take the new center and remove the corresponding value from the other List
                    centers.add(P.remove(j-1));
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

    //time complexity: O(|P|*k)
    private static double kmeansObj(ArrayList<Vector> P, ArrayList<Vector> C) {
        //Arraylist that keeps the distances between a point and all centers
        ArrayList<Double> distancePC = new ArrayList<>();
        //ArrayList to keep the distance of a point from its center (minimum)
        ArrayList<Double> dists = new ArrayList<>();
        for (Vector point : P) {
            for (Vector center : C) {
                //writing in distancePC the squared distances between a point and all centers
                distancePC.add(Vectors.sqdist(point, center));
            }
            //adding in dists the minimum distance -> the distance of a point from its center
            dists.add(Collections.min(distancePC));
            //clearing of distancePC, for the next point
            distancePC.clear();
        }
        //now i have dists with all distances between a point and its center
        //computing the average
        double sum = 0;
        for (Double dist : dists) {
            sum += dist;
        }
        return sum/P.size();
    }
}
