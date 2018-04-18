package it.unipd.dei.bdc1718;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class G08HM3 {

  public static void main(String[] args) throws FileNotFoundException {
    if (args.length == 0){
      throw new IllegalArgumentException("Expecting the file name on the command line");
    }

    //HOMEWORK 3

    // Setup Spark
    SparkConf conf = new SparkConf(true).setAppName("Preliminaries");
    JavaSparkContext sc = new JavaSparkContext(conf);

  }
}
