# Checking viability of LSH with One versus All planes

import tensorflow as tf
import numpy as np
import random
import math
from sklearn import svm
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../testing-data/MNIST_data/",
  one_hot=True)

# Hardware Specifications
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Graph Constants
size = [28, 28, 1]
nKernels = [8, 16, 32]
fully_connected_nodes = 128
poolS = 2

 # LSH Testing
batchS = 32
nPlanes = 100 

nClasses = 10
nSuppImgs = 5 
nSupportTraining = 10000
nTrials = 1000
nSupp = nClasses * nSuppImgs

numbers = [1, 2, 3, 4, 5]
excluded_numbers = []
while len(numbers) < nClasses:
  selected_class = random.randint(0, 9)
  while selected_class in numbers or selected_class in excluded_numbers:
    selected_class = random.randint(0, 9)
  numbers.append(selected_class)

if len(sys.argv) < 2:
  print("no model provided")
  exit()
SAVE_PATH = sys.argv[1]

tf.reset_default_graph()

# Query Information - vector
dataset = tf.placeholder(tf.float32, [None]+size)    # batch size

# Network Function, call for each image to create feature vector
def create_network(img, size, First = False):
  currInp = img
  layer = 0
  currFilt = size[2]
  
  for k in nKernels:
    with tf.variable_scope('conv' + 
      str(layer),reuse=tf.AUTO_REUSE) as varscope:
      layer += 1
      weight = tf.get_variable('weight', [3,3,currFilt,k])
      currFilt = k
      bias = tf.get_variable('bias', [k], initializer = 
        tf.constant_initializer(0.0))
      convR = tf.nn.conv2d(currInp, weight, strides=[1,1,1,1],
        padding= "SAME")
      convR = tf.add(convR, bias)
      reluR = tf.nn.relu(convR)
      poolR = tf.nn.max_pool(reluR, ksize=[1,2,2,1], strides=[1,2,2,1], 
        padding="SAME")
      currInp = poolR
  
  return currInp

features = create_network(dataset, size)

# Cosine Similarity 
def cos_similarities(supports, query): 
  dotProduct = np.sum(np.multiply(supports, query), (1))
  supportsMagn = np.sqrt(np.sum(np.square(supports), (1)))
  cosDist = dotProduct / np.clip(supportsMagn, 1e-10, float("inf"))
  return cosDist

# LSH Functions

# Generates a matrix ("tensor") of dimension nKernels[-1] (which is the 
# size of the feature vector our network outputs) by the number of planes.
# Using SVM mechanism in order to develop planes to seaprate each plane
# from the rest
def gen_lsh_pick_planes(num_planes, feature_vectors, labels):
  
  lsh_matrix = []
  lsh_offset_vals = []
  
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
        
    # Decrease C if the data turns out (very) noisy    
    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(x,y)
      
    lsh_matrix.append(clf.coef_[0])
    
    # create offset value
    temp_vec_is_set = False
    # number of dimensions of the space
    temp_vec = [0]*len(feature_vectors[0])
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
  lsh_vectors = np.matmul(feature_vectors, np.transpose(LSH_matrix))
  lsh_vectors = np.subtract(lsh_vectors, lsh_offset_vals)
  lsh_bin = np.sign(lsh_vectors)
  lsh_bin = np.clip(lsh_bin, 0, 1)
  return lsh_bin, lsh_vectors

# Generate distance
def lsh_dist(lshSupp, lshQueryO, lshVecSupp, lshVecQuery):
  qlist = []
  lshQuery = np.empty([nSupp, 10])
  lshQuery2 = np.empty([nSupp, 10])
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
 
  # for these, we may want to just feed through all of the mnist data
  rawDataset = np.reshape(mnist.train.images, [mnist.train.images.shape[0]]
    + size)
  rawLabels = mnist.train.labels

  featureVectors = None
  for i in range(int(len(rawDataset)/1000)):
    FEAT = (session.run([features], feed_dict =
      {dataset: rawDataset[i*1000:(i+1)*1000]}))
    FEAT = np.asarray(FEAT)
    if featureVectors is None:
      featureVectors = np.empty([len(rawDataset), FEAT.shape[2], FEAT.shape[3], FEAT.shape[4]])
    featureVectors[i*1000:(i+1)*1000] = FEAT[0]
  featureVectors = np.reshape(featureVectors, (len(rawDataset), -1))  

  queryDataset = np.reshape(mnist.test.images, 
    [mnist.test.images.shape[0]]+size)
  queryLabels = mnist.test.labels
  
  queryFeatureVectors = None
  for i in range(int(len(queryDataset)/1000)):
    FEAT = (session.run([features], feed_dict =
      {dataset: queryDataset[i*1000:(i+1)*1000]}))
    FEAT = np.asarray(FEAT)
    if queryFeatureVectors is None:
      queryFeatureVectors = np.empty([len(queryDataset), FEAT.shape[2], FEAT.shape[3], FEAT.shape[4]])
    queryFeatureVectors[i*1000:(i+1)*1000] = FEAT[0]
  queryFeatureVectors = np.reshape(queryFeatureVectors, (len(queryDataset), -1))

lsh_planes, lsh_offset_vals = gen_lsh_pick_planes(nPlanes, 
  featureVectors[:10000], rawLabels)

sumEff = 0

cos_acc = 0
lsh_acc = 0
lsh_acc2 = 0
# choose random support vectors
supp = []
supp_labels = []
for j in numbers: 
  n = 0 
  while n < nSuppImgs:
    supp_index = random.randint(0, mnist.train.images.shape[0] - 1)
    while int(np.argmax(mnist.train.labels[supp_index])) != j:
      supp_index += 1
      if supp_index == len(mnist.train.images):
        supp_index = 0 
    n += 1   
    supp.append(featureVectors[supp_index])
    supp_labels.append(int(np.argmax(mnist.train.labels[supp_index])))


for i in range(nTrials):
  
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

  if LSHMatch2 == query_label:
    lsh_acc2+=1   

print("Cos Acc: "+str(float(cos_acc)/(nTrials) * 100)+"%")
print("LSH Acc: "+str(float(lsh_acc)/(nTrials) * 100)+"%")
print("LSH Acc2: "+str(float(lsh_acc2)/(nTrials) * 100)+"%")
eff = float(sumEff) / nTrials
print("Effectiveness: "+str(eff))
