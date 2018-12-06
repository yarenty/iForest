package com.yarenty.h2o

import java.net.URI

import hex.tree.isofor.IsolationForest
import hex.tree.isofor.IsolationForestModel.IsolationForestParameters
import water.H2OStarter
import water.fvec.H2OFrame
import water.util.Log

object DoAnomaly extends App {

  H2OStarter.start(Array("-name", "YOLO", "--ga_opt_out"), System.getProperty("user.dir"))


  println("ready to GO")

  Log.info("Ready to GO!")

  Log.info("load data")
  
 val train = new H2OFrame(new URI("/opt/data/demo/EH_BL1375B/data2.csv"))
  
  val params = new IsolationForestParameters()
    params._train = train.key


  params._mtries = -1
  params._sample_size = 12
  params._max_depth = 3 // log2(_sample_size) = 8

  params._sample_rate = -1
  params._min_rows = 1
  params._min_split_improvement = 0
  params._nbins = 2
  params._nbins_cats = 2
  
  params._ntrees = 100
  
  val iForest = new IsolationForest(params)

  val iModel = iForest.trainModel.get
  
  val predicitons = iModel.score(train)
  
  val summary = train.add(predicitons)


//  println(iModel.getMostImportantFeatures(10).mkString(";\n"))
  
}
