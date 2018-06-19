# Checking viability of LSH with random planes

import tensorflow as tf
import numpy as np
import random
import math

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)


# Graph Constants
size = [28, 28, 1]
nKernels = [8, 16, 32]
fully_connected_nodes = 128
poolS = 2

# LSH Testing
nSupp = 30
nSupportTraining = 10000
MAX_SLOPE = 10
nPlanes = 500
nTrials = 1000

SAVE_PATH = "model/lsh-training-cosine-3-5000"

tf.reset_default_graph()

# Query Information - Vector of images
dataset = tf.placeholder(tf.float32, [None]+size)

# Network Function, call for each image to create feature vector
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
      convR = tf.nn.conv2d(currInp, weight, strides=[1,1,1,1], padding="SAME")
      convR = tf.add(convR, bias)
      reluR = tf.nn.relu(convR)
      poolR = tf.nn.max_pool(reluR, ksize=[1,2,2,1], strides=[1,2,2,1], 
        padding="SAME")
      currInp = poolR
  
  with tf.variable_scope('FC', reuse = tf.AUTO_REUSE) as varscope:
    CurrentShape=currInp.get_shape()
    FeatureLength = int(CurrentShape[1]*CurrentShape[2]*CurrentShape[3])
    FC = tf.reshape(currInp, [-1,FeatureLength])
    W = tf.get_variable('W',[FeatureLength,fully_connected_nodes])
    FC = tf.matmul(FC, W)
    Bias = tf.get_variable('Bias',[fully_connected_nodes])
    FC = tf.add(FC, Bias)
    FC = tf.reshape(FC, [-1,fully_connected_nodes])
  
  return FC

features = create_network(dataset, size)

# Cosine Similarity Function
def cos_similarities(supports, query): 
  dotProduct = np.sum(np.multiply(supports, query), (1))
  supportsMagn = np.sqrt(np.sum(np.square(supports), (1)))
  cosDist = dotProduct / np.clip(supportsMagn, 1e-10, float("inf"))
  return cosDist

# LSH Function

# Generates a matrix ("tensor") of dimension nKernels[-1] (which is the 
# size of the feature vector our network outputs) by the number of planes.
# Random planes with slopes between -1 and 1
def gen_lsh_pick_planes(num_planes, feature_vectors, labels):
  return np.transpose((np.matlib.rand(fully_connected_nodes, num_planes) 
    - 0.5) * 2), np.zeros(num_planes)
  
    # lsh_offset_vals.append(np.dot(clf.intercept_[0], clf.coef_[0]))
  return lsh_matrix, lsh_offset_vals, clfs

# Calculate Hash for the feature vector
def lsh_hash(feature_vectors, LSH_matrix, lsh_offset_vals):
  feature_vectors = np.reshape(feature_vectors, (-1, fully_connected_nodes))
  lsh_vectors = np.matmul(feature_vectors, np.transpose(LSH_matrix))
  lsh_vectors = np.subtract(lsh_vectors, lsh_offset_vals)
  lsh_bin = np.sign(lsh_vectors)
  lsh_bin = np.clip(lsh_bin, 0, 1)
  return lsh_bin, lsh_vectors

# Generate Distance
def lsh_dist(lshSupp, lshQueryO, lshVecSupp, lshVecQuery):
  qlist = []
  lshQuery = np.empty([nSupp, nPlanes])
  lshQuery2 = np.empty([nSupp, nPlanes])
  for i in range(nSupp):
    lshQuery[i] = lshQueryO
    lshQuery2[i] = lshVecQuery
  dist = np.equal(lshSupp, lshQuery)
  dist = np.sum(dist.astype(int), 1)
  dist_2 = np.multiply(lshVecSupp, lshQuery2)
  dist2 = np.divide(1.0, np.add(1.0, np.exp(np.multiply(-50.0, dist_2))))
  dist2 = np.sum(dist2, 1)  # check this!
  return dist, dist2

# Session

init = tf.global_variables_initializer()

with tf.Session() as session:
  session.run(init)
  Saver = tf.train.Saver()
  Saver.restore(session, SAVE_PATH)
 
  # for these, we may want to just feed through all of the mnist data for the 
  # feature vectors
  rawDataset = np.reshape(mnist.train.images, [mnist.train.images.shape[0]] + 
    size)
  rawLabels = mnist.train.labels
  
  featureVectors = np.empty([55000, fully_connected_nodes])
  for i in range(55):
    FEAT = (session.run([features], feed_dict = 
      {dataset: rawDataset[i*1000:(i+1)*1000]}))
    FEAT = np.asarray(FEAT)
    featureVectors[i*1000:(i+1)*1000] = FEAT[0]

  queryDataset = np.reshape(mnist.test.images, 
    [mnist.test.images.shape[0]]+size)
  queryLabels = mnist.test.labels
  
  queryFeatureVectors = np.empty([10000, fully_connected_nodes])
  for i in range(10):
    FEAT = (session.run([features], feed_dict = 
      {dataset: queryDataset[i*1000:(i+1)*1000]}))
    FEAT = np.asarray(FEAT)
    queryFeatureVectors[i*1000:(i+1)*1000] = FEAT[0]

lsh_planes, lsh_offset_vals = gen_lsh_pick_planes(nPlanes, 
  featureVectors[:nSupportTraining], rawLabels)

sumEff = 0

cos_acc = 0
lsh_acc = 0
lsh_acc2 = 0
for i in range(nTrials):
  
  # choose random support vectors
  supp = []
  supp_labels = []
  occurences = np.zeros(10)
  for j in range(nSupp):
    supp_index = random.randint(0, mnist.train.images.shape[0] - 1)
    while occurences[np.argmax(mnist.train.labels[supp_index])] > 0 
      and 0 in occurences:
      supp_index = random.randint(0, mnist.train.images.shape[0] - 1)
    supp.append(featureVectors[supp_index])
    supp_labels.append(np.argmax(mnist.train.labels[supp_index]))
    occurences[np.argmax(mnist.train.labels[supp_index])] += 1
  
  # choose random query
  query_index = random.randint(0, mnist.test.images.shape[0] - 1)
  query = queryFeatureVectors[query_index]
  query_label = np.argmax(queryLabels[query_index])
  
  # get lsh binaries (from application to matrix) for supp and query
  lsh_bin, lsh_vec = lsh_hash(np.asarray(supp), lsh_planes, lsh_offset_vals)
  lsh_bin_q, lsh_vec_q = lsh_hash(np.asarray(query), lsh_planes,
    lsh_offset_vals)

  # calculate lsh distances
  # find closest match
  distances, distances2 = lsh_dist(lsh_bin, lsh_bin_q, lsh_vec, lsh_vec_q)
  maximum = max(distances)
  LSHMatch = supp_labels[np.argmax(distances)]
  LSHMatch2 = supp_labels[np.argmax(distances2)]

  # Create list of queries for comparison
  q_list = []
  for j in range(nSupp):
    q_list.append(query)
  q_list = np.asarray(q_list)
  
  # find closest match with cosine
  cosDistances = cos_similarities(supp, q_list)
  cosMatch = supp_labels[np.argmax(cosDistances)]
 
  if cosMatch == query_label:
    cos_acc += 1
  
  if cosMatch == LSHMatch:
    sumEff += 1
  
  if LSHMatch == query_label:
    lsh_acc+=1

  if LSHMatch2 == query_label:
    lsh_acc2+=1   

print("Cos Acc: "+str(float(cos_acc)/(nTrials) * 100)+"%")
print("LSH Acc: "+str(float(lsh_acc)/(nTrials) * 100)+"%")
print("LSH Acc2: "+str(float(lsh_acc2)/(nTrials) * 100)+"%")
eff = float(sumEff) / nTrials
print("Effectiveness: "+str(eff))