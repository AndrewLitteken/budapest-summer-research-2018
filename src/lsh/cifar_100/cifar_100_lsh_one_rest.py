# Checking viability of LSH with One versus All planes CIFAR-100

from sklearn import svm
import tensorflow as tf
import numpy as np
import getopt
import random
import math
import sys
import os
import pickle

import cifar_100

cifar_100.get_data()

train_file_path = "../../../testing-data/cifar-100/train"
train_images_raw = np.empty((0, 3072))
train_labels_raw = np.empty((0))
with open(train_file_path, 'rb') as cifar_file:
  data = pickle.load(cifar_file, encoding = 'bytes')
  train_images_raw = data[b"data"]
  train_labels_raw = data[b"fine_labels"]

test_file_name = "../../../testing-data/cifar-100/test"
with open(test_file_name, 'rb') as cifar_file:
  test = pickle.load(cifar_file, encoding = 'bytes')

# Hardware Specifications
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Graph Constants
size = [32, 32, 3]
nKernels = [64, 64, 64]
fully_connected_nodes = 128
poolS = 2

#Training information
batchS = 32
nPlanes = 100

nClasses = 3
nSuppImgs = 5
nSupportTraining = 10000
nTrials = 1000
unseen = False

opts, args = getopt.getopt(sys.argv[1:], "hc:i:s:u", ["help", 
  "num_classes=", "num_supports=", "num_iterations=",
  "unseen"])

for o, a in opts:
  if o in ("-c", "--num_classes"):
    nClasses = int(a)
  elif o in ("-s", "--num_supports"):
    nImgsSuppClass = int(a)
  elif o in ("-i", "--num_iterations"):
    nTrials = int(a)
  elif o in ("-h", "--help"):
    help_message()
  elif o in ("-u", "--unseen"):
    unseen = True
  else:
    print("unhandled option")
    help_message()

if len(args) < 1:
  print("no model provided")
  exit(1)
SAVE_PATH = args[0]

end_file = SAVE_PATH.split("/")[-1]

end_file = end_file.split("-")

index = 0
reference_dict = None
while not reference_dict:
  if end_file[index] == "cosine":
    reference_dict = (("classes", end_file[index + 1]),
                      ("supports", end_file[index+2]))
  elif end_file[index] == "lsh":
    if end_file[index + 1] == "one":
      reference_dict = (("classes", end_file[index + 4]),
                        ("supports", end_file[index + 5],
                        ("period", end_file[index + 3]),))
    else:
      reference_dict = (("classes", end_file[index + 3]),
                        ("supports", end_file[index + 4]),
                        ("planes", end_file[index + 2]),
                        ("training", end_file[index + 5]))
  index+=1

nSupp = nClasses * nSuppImgs

train_images = []
train_labels = []
list_range = np.arange(len(train_images_raw))
np.random.shuffle(list_range)
for index, i in enumerate(list_range):
  if ((train_labels_raw[i] < 80 and not unseen) or (train_labels_raw[i]
    > 79 and unseen)):
    train_images.append(train_images_raw[i])
    train_labels.append(train_labels_raw[i])


train_images = np.reshape(train_images, [len(train_images), 3, 32, 32])
train_images = np.transpose(train_images, [0, 2, 3, 1]) 

test_images_raw = test[b"data"]
test_labels_raw = test[b"fine_labels"]
list_range = np.arange(len(test_images_raw))
np.random.shuffle(list_range)

test_images = []
test_labels = []
for index, i in enumerate(list_range):
  if ((test_labels_raw[i] < 80 and not unseen) or (test_labels_raw[i]
    > 79 and unseen)):
    test_images.append(test_images_raw[i])
    test_labels.append(test_labels_raw[i])
test_images = np.reshape(test_images, [len(test_images), 3, 32, 32])
test_images = np.transpose(test_images, [0, 2, 3, 1])

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
def gen_lsh_pick_planes(images, feature_vectors, labels):
  
  lsh_matrix = []
  lsh_offset_vals = []
  
  feature_vectors = np.reshape(np.asarray(feature_vectors),
    (len(feature_vectors), -1))

  for index_i, i in enumerate(images):
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
    
    temp_mul = np.matmul(np.asarray(temp_vec), lsh_matrix[index_i])
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
 
  # for these, we may want to just feed through all of the mnist data for 
  # the feature vectors
  rawDataset = train_images
  rawLabels = train_labels

  featureVectors = None
  for i in range(int(len(rawDataset)/1000)):
    FEAT = (session.run([features], feed_dict =
      {dataset: rawDataset[i*1000:(i+1)*1000]}))
    FEAT = np.asarray(FEAT)
    if featureVectors is None:
      featureVectors = np.empty([len(rawDataset), FEAT.shape[2], FEAT.shape[3], FEAT.shape[4]])
    featureVectors[i*1000:(i+1)*1000] = FEAT[0]
  featureVectors = np.reshape(featureVectors, (len(rawDataset), -1))

  queryDataset = test_images
  queryLabels = test_labels

  queryFeatureVectors = None
  for i in range(int(len(queryDataset)/1000)):
    FEAT = (session.run([features], feed_dict =
      {dataset: queryDataset[i*1000:(i+1)*1000]}))
    FEAT = np.asarray(FEAT)
    if queryFeatureVectors is None:
      queryFeatureVectors = np.empty([len(queryDataset), FEAT.shape[2], FEAT.shape[3], FEAT.shape[4]])
    queryFeatureVectors[i*1000:(i+1)*1000] = FEAT[0]
  queryFeatureVectors = np.reshape(queryFeatureVectors, (len(queryDataset), -1))

sumEff = 0

cos_acc = 0
lsh_acc = 0
lsh_acc2 = 0

supp = []
supp_labels = []
images = []
while len(images) < nClasses:
  choice = random.choice(train_labels)
  while choice in images:
    choice = random.choice(train_labels)
  n = 0
  while n < nSuppImgs:
    supp_index = random.randint(0, train_images.shape[0] - 1)
    while int(train_labels[supp_index]) != choice:
      supp_index += 1
      if supp_index == len(train_images):
        supp_index = 0
    n += 1
    supp.append(featureVectors[supp_index])
    supp_labels.append(int(train_labels[supp_index]))
  images.append(choice)

lsh_planes, lsh_offset_vals = gen_lsh_pick_planes(images, 
  supp, supp_labels)

for i in range(nTrials):
  
  # choose random query
  query_value = random.choice(images)
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
#print("Cos Acc: "+str(float(cos_acc)/(nTrials) * 100)+"%")
#print("LSH Acc: "+str(float(lsh_acc)/(nTrials) * 100)+"%")
#print("LSH Acc2: "+str(float(lsh_acc2)/(nTrials) * 100)+"%")
cos_lsh_acc = float(cos_acc)/(nTrials)
calc_lsh_acc = float(lsh_acc)/(nTrials)
calc_lsh_acc2 = float(lsh_acc2)/(nTrials)
eff = float(sumEff) / nTrials
#print("Effectiveness: "+str(eff))
output="lsh_one_rest,"
for i in reference_dict:
  output += i[1] + ","
output += str(cos_lsh_acc) + "," + str(calc_lsh_acc) + "," + str(calc_lsh_acc2)
print(output)
