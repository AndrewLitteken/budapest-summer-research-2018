# Checking viability of LSH with One versus All planes

import tensorflow as tf
import numpy as np
import random
import math
from sklearn import svm
import sys
import os
import pickle

train_file_path = "../../../testing-data/cifar/data_batch_"
train_images_raw = np.empty((0, 3072))
train_labels_raw = np.empty((0))
for i in range(1,6):
  train_file_name = train_file_path + str(i)
  with open(train_file_name, 'rb') as cifar_file:
    data = pickle.load(cifar_file, encoding = 'bytes')
    train_images_raw = np.concatenate((train_images_raw, data[b"data"]), 
      axis = 0)
    train_labels_raw = np.concatenate((train_labels_raw, data[b"labels"]), 
      axis = 0)

test_file_name = "../../../testing-data/cifar/test_batch"
with open(test_file_name, 'rb') as cifar_file:
  test = pickle.load(cifar_file, encoding = 'bytes')

# Hardware Specifications
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Graph Constants
size = [32, 32, 3]
nKernels = [8, 16, 32]
fully_connected_nodes = 128
poolS = 2

#Training information
batchS = 32
nPlanes = 100

nClasses = 10
nSuppImgs = 5
nSupportTraining = 10000
nTrials = 1000
nSupp = nClasses * nSuppImgs

numbers = []
while len(numbers) < nClasses:
  selected_class = random.randint(0, 9)
  while selected_class in numbers:
    selected_class = random.randint(0, 9)
  numbers.append(selected_class)

if len(sys.argv) < 2:
  print("no model provided")
  exit()
SAVE_PATH = sys.argv[1]

train_images = []
train_labels = []
list_range = np.arange(len(train_images_raw))
np.random.shuffle(list_range)
for index, i in enumerate(list_range):
  train_images.append(train_images_raw[i])
  train_labels.append(train_labels_raw[i])

train_images = np.reshape(train_images, [len(train_images)] + size)

test_images = test[b"data"]
test_images = np.reshape(test_images, [len(test_images)] + size)
test_labels = test[b"labels"]

tf.reset_default_graph()

# Query Information - Vector of images
dataset = tf.placeholder(tf.float32, [None]+size)

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

  for index_i, i in enumerate(numbers):
    x = []
    y = []
    for index in range(len(feature_vectors)):
      # print(np.asarray(feature_vectors[index]).shape)
      x.append(feature_vectors[index])
      
      # create a vector y which classifies each vector as "i" or "not i"
      if labels[index] == i:
        y.append(1)
      else:
        y.append(0)
        
    # Decrease C if the data turns out (very) noisy
    print("attempting to fit data")
    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(x,y)
    print("data fit")
      
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
    
    temp_mul = np.matmul(np.asarray(temp_vec), lsh_matrix[index_i])
    lsh_offset_vals.append(temp_mul)

  return lsh_matrix, lsh_offset_vals

def lsh_hash(feature_vectors, LSH_matrix, lsh_offset_vals):
  feature_vectors = np.reshape(feature_vectors, (-1, fully_connected_nodes))
  lsh_vectors = np.matmul(feature_vectors, np.transpose(LSH_matrix))
  lsh_vectors = np.subtract(lsh_vectors, lsh_offset_vals)
  lsh_bin = np.sign(lsh_vectors)
  lsh_bin = np.clip(lsh_bin, 0, 1)
  return lsh_bin, lsh_vectors

# Generate distance
def lsh_dist(lshSupp, lshQueryO, lshVecSupp, lshVecQuery):
  qlist = []
  lshQuery = np.empty([nSupp, nClasses])
  lshQuery2 = np.empty([nSupp, nClasses])
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
  rawDataset = np.reshape(train_images, [train_images.shape[0]] 
    + [3, 32, 32])
  rawDataset = np.transpose(rawDataset, [0, 2, 3, 1])
  rawLabels = train_labels
  
  featureVectors = np.empty([len(rawDataset), fully_connected_nodes])
  for i in range(int(len(rawDataset)/1000)):
    FEAT = (session.run([features], feed_dict = 
      {dataset: rawDataset[i*1000:(i+1)*1000]}))
    FEAT = np.asarray(FEAT)
    featureVectors[i*1000:(i+1)*1000] = FEAT[0]

  queryDataset = np.reshape(test_images, [test_images.shape[0]]
    + [3, 32, 32])
  queryDataset = np.transpose(queryDataset, [0, 2, 3, 1])
  queryLabels = test_labels
  
  queryFeatureVectors = np.empty([len(queryDataset), fully_connected_nodes])
  for i in range(int(len(queryDataset)/1000)):
    FEAT = (session.run([features], feed_dict = 
      {dataset: queryDataset[i*1000:(i+1)*1000]}))
    FEAT = np.asarray(FEAT)
    queryFeatureVectors[i*1000:(i+1)*1000] = FEAT[0]

#lsh_planes, lsh_offset_vals = gen_lsh_pick_planes(nPlanes, 
#  featureVectors[:100], rawLabels)

sumEff = 0

cos_acc = 0
lsh_acc = 0
lsh_acc2 = 0
supp = []
supp_labels = []
for j in numbers:
  n = 0
  while n < nSuppImgs:
    supp_index = random.randint(0, train_images.shape[0] - 1)
    while int(train_labels[supp_index]) != j:
      supp_index += 1
      if supp_index == len(train_images):
        supp_index = 0
    n += 1
    supp.append(featureVectors[supp_index])
    supp_labels.append(int(train_labels[supp_index]))

lsh_planes, lsh_offset_vals = gen_lsh_pick_planes(nPlanes, 
  supp, supp_labels)

for i in range(nTrials):
  
  # choose random support vectors
  
  # choose random query
  query_value = random.choice(numbers)
  query_index = random.randint(0, test_images.shape[0] - 1)
  while query_value != int(queryLabels[query_index]):
    query_index += 1
    if query_index == len(test_images):
      query_index = 0
  query = queryFeatureVectors[query_index]
  query_label = queryLabels[query_index]

  
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
