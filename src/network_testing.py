# Code to test many networks at once

import tensorflow as tf
import numpy as np
import random
import math
import sys
import os
from sklearn import svm

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Hardware Specifications
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def create_network(img, size, First = False):
  currInp = img
  layer = 0
  currFilt = size[2]
  
  for k in nKernels:
    with tf.variable_scope('conv'+str(layer),reuse=tf.AUTO_REUSE) as varscope:
      layer += 1
      weight = tf.get_variable('weight', [3,3,currFilt,k])   # make parameters!
      currFilt = k
      bias = tf.get_variable('bias', [k], initializer =
        tf.constant_initializer(0.0))
      convR = tf.nn.conv2d(currInp, weight, strides = [1,1,1,1],
        padding="SAME")
      convR = tf.add(convR, bias)
      reluR = tf.nn.relu(convR)
      poolR = tf.nn.max_pool(reluR, ksize = [1,2,2,1], strides = [1,2,2,1], 
        padding="SAME")
      currInp = poolR
  
  with tf.variable_scope('FC', reuse = tf.AUTO_REUSE) as varscope:
    CurrentShape=currInp.get_shape()
    FeatureLength = int(CurrentShape[1]*CurrentShape[2]*CurrentShape[3])
    FC = tf.reshape(currInp, [-1,FeatureLength])
    W = tf.get_variable('W',[FeatureLength,128])
    FC = tf.matmul(FC, W)
    Bias = tf.get_variable('Bias',[128])
    FC = tf.add(FC, Bias)
    FC = tf.reshape(FC, [-1,128])
  
  return FC

# Cosine Similarity
def cos_similarities(supports, query): 
  dotProduct = np.sum(np.multiply(supports, query), (1))
  supportsMagn = np.sqrt(np.sum(np.square(supports), (1)))
  cosDist = dotProduct / np.clip(supportsMagn, 1e-10, float("inf"))
  return cosDist

"""###**LSH Functions**"""

def gen_lsh(num_planes):
  return np.transpose((np.matlib.rand(128, num_planes) - 0.5) * 2), None

# Generates a matrix ("tensor") of dimension nKernels[-1] (which is the 
# size of the feature vector our network outputs) by the number of planes.
def gen_lsh_pick_planes(num_planes, feature_vectors, labels):
  
  lsh_matrix = []
  lsh_offset_vals = []
  clfs = []
  
  feature_vectors = np.reshape(np.asarray(feature_vectors),
    (len(feature_vectors), -1))

  for i in range(10):
    x = []
    y = []
    current_label = [0.] * 10
    current_label[i] = 1.
    for index in range(len(feature_vectors)):
      # print(np.asarray(feature_vectors[index]).shape)
      x.append(feature_vectors[index])
      
      # create a vector y which classifies each vector as "i" or "not i"
      if np.array_equal(labels[index], current_label):
        y.append(1)
      else:
        y.append(0)
      
    
    clfs.append(clf)
    lsh_matrix.append(clf.coef_[0])
    
    temp_vec_is_set = False
    temp_vec = [0]*len(feature_vectors[0])  # number of dimensions of the space
    for j in range(0, len(feature_vectors[0])):
      # if never enters this if statement, that is an error
      if clf.coef_[0][j] != 0 and not temp_vec_is_set:
        temp_vec[j] = -1*clf.intercept_[0] / clf.coef_[0][j]
        temp_vec_is_set = True
        break 
     
    if (not temp_vec_is_set): 
      print("BAD. Temp_vec not set, which doesn't make sense.")
    
    temp_mul = np.matmul(np.asarray(temp_vec), lsh_matrix[i])
    lsh_offset_vals.append(temp_mul)

  return lsh_matrix, lsh_offset_vals

def lsh_hash(feature_vectors, LSH_matrix, lsh_offset_vals):
  feature_vectors = np.reshape(feature_vectors, (-1, 128))
  lsh_vectors = np.matmul(feature_vectors, np.transpose(LSH_matrix))
  if lsh_offset_vals:
    lsh_vectors = np.subtract(lsh_vectors, lsh_offset_vals)
  lsh_bin = np.sign(lsh_vectors)
  lsh_bin = np.clip(lsh_bin, 0, 1)
  return lsh_bin

def lsh_dist(lshSupp, lshQueryO):
  qlist = []
  lshQuery = np.empty([nSupp, nPlanes])
  for i in range(nSupp):
    lshQuery[i] = lshQueryO
  dist = np.equal(lshSupp, lshQuery)
  dist = np.sum(dist.astype(int), 1)  # check this!
  return dist


# Graph features
size = [28, 28, 1]
nKernels = [8, 16, 32]
poolS = 2

nSupp = 30
MAX_SLOPE = 10

# LSH setup
nPlanes = 500
nTrials = 1000

save_paths = []

if len(sys.argv) < 2:
  save_paths.append("./model/lsh-training-cosine-3-5000")
else:
  if sys.argv[1] == "random_planes":
    trained_list = ["False", "True"]
    nPlanes_list = [10, 20, 100, 200, 1000]

    base = "./model/lsh-training-random-"
    for item in trained_list:
      for plane in nPlanes_list:
        save_paths.append(base+str(plane)+"-3-"+str(item)+"-5000")

  elif sys.argv[1] == "trained_planes":
    periods = [100, 200, 500, 1000]

    base = "./model/lsh-training-one-all-"
    for item in periods:
      save_paths.append(base+str(item)+"-3-5000")

for SAVE_PATH in save_paths:

  tf.reset_default_graph()
  # Query Information - vector
  dataset = tf.placeholder(tf.float32, [None]+size)    # batch size
  
  features = create_network(dataset, size)

  # Session
  init = tf.global_variables_initializer()
  with tf.Session() as session:
    session.run(init)

    Saver = tf.train.Saver()
    Saver.restore(session, SAVE_PATH)

    # for these, we may want to just feed through all of the mnist data
    rawDataset = np.reshape(mnist.train.images, 
      [mnist.train.images.shape[0]]+size)
    rawLabels = mnist.train.labels

    featureVectors = np.empty([55000, 128])
    for i in range(55):
      FEAT = (session.run([features], feed_dict = 
        {dataset: rawDataset[i*1000:(i+1)*1000]}))
      FEAT = np.asarray(FEAT)
      featureVectors[i*1000:(i+1)*1000] = FEAT[0]

    queryDataset = np.reshape(mnist.test.images, 
      [mnist.test.images.shape[0]]+size)
    queryLabels = mnist.test.labels
  
    queryFeatureVectors = np.empty([10000, 128])
    for i in range(10):
      FEAT = (session.run([features], feed_dict = 
        {dataset: queryDataset[i*1000:(i+1)*1000]}))
      FEAT = np.asarray(FEAT)
      queryFeatureVectors[i*1000:(i+1)*1000] = FEAT[0]

  methods = ["trained", "random"]
  for lsh_method in methods:
    if lsh_method == "trained":
      nPlanes = 10
      lsh_planes, lsh_offset_vals, clfs = gen_lsh_pick_planes(nPlanes, 
        featureVectors[:5000], rawLabels)
    elif lsh_method == "random":
      lsh_planes, lsh_offset_vals = gen_lsh(nPlanes)

    sumEff = 0
    cos_acc = 0
    lsh_acc = 0
    for i in range(nTrials):

      # choose random support vectors
      supp = []
      supp_labels = []
      occurences = np.zeros(10)
      for j in range(nSupp):
        supp_index = random.randint(0, mnist.train.images.shape[0] - 1)
        supp.append(featureVectors[supp_index])
        supp_labels.append(np.argmax(mnist.train.labels[supp_index]))
        occurences[np.argmax(mnist.train.labels[supp_index])] += 1
  
      # choose random query
      query_index = random.randint(0, mnist.test.images.shape[0] - 1)
      query = queryFeatureVectors[query_index]
      query_label = np.argmax(queryLabels[query_index])
  
      # get lsh binaries (from application to matrix) for supp and query
      lsh_bin = lsh_hash(np.asarray(supp), lsh_planes, lsh_offset_vals)
      lsh_bin_q = lsh_hash(np.asarray(query), lsh_planes, lsh_offset_vals)

      # calculate lsh distances
      # find closest match
      distances = lsh_dist(lsh_bin, lsh_bin_q)
      maximum = max(distances)

      q_list = []
      for j in range(nSupp):
        q_list.append(query)
      q_list = np.asarray(q_list)
  
      # find closest match
      cosDistances = cos_similarities(supp, q_list)
      cosMatch = supp_labels[np.argmax(cosDistances)]
 
      if cosMatch == query_label:
        cos_acc += 1
  
      if cosMatch == LSHMatch:
        sumEff += 1
  
      if LSHMatch == query_label:
        lsh_acc+=1

    cos_acc_val = str(float(cos_acc)/(nTrials))
    lsh_acc_val = str(float(lsh_acc)/(nTrials))
    eff = str(float(sumEff) / nTrials)
    categories = SAVE_PATH.split("-")
    training = categories[2]
    if training == "random":
      n_Planes = categories[3]
      n_Classes = categories[4]
      trainable = categories[5]
      print(training+","+n_Planes+","+n_Classes+","+trainable+","+
        lsh_method+","+cos_acc_val+","+lsh_acc_val+","+eff)
    elif training == "one":
      period = categories[4]
      n_Classes = categories[5]
      print(training+","+period+","+n_Classes+","+lsh_method+","+
        cos_acc_val+","+lsh_acc_val+","+eff)
    else:
      print(lsh_method+","+cos_acc_val+","+lsh_acc_val+","+eff)
      
