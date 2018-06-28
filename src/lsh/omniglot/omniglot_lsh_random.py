# Checking viability of LSH with random planes Omniglot

from skimage import transform, io
import tensorflow as tf
import numpy as np
import scipy.misc
import getopt
import random
import math
import sys
import os

train_file_path = "../../../testing-data/omniglot/"

# Hardware Specifications
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def make_dir_list(data_dir):
  path_train = "{}images_background/".format(data_dir)
  path_test = "{}images_evaluation/".format(data_dir)

  train_dirs = []
  test_dirs = []
  for alphabet in os.listdir(path_train):
    if not alphabet.startswith('.') : 
      for character in os.listdir("{}{}/".format(path_train,alphabet)):
        train_dirs.append("{}{}/{}".format(path_train, alphabet, character))

  for alphabet in os.listdir(path_test):
    if not alphabet.startswith('.') : 
      for character in os.listdir("{}{}/".format(path_test, alphabet)):
        test_dirs.append("{}{}/{}".format(path_test, alphabet, character))

  return np.asarray(train_dirs), np.asarray(test_dirs)

# Graph Constants
size = [28, 28, 1]
nKernels = [64, 64, 64]
fully_connected_nodes = 128
poolS = 2

def get_images():
  train_image_list, test_image_list = make_dir_list(train_file_path)

  raw_train_images = []
  raw_train_labels = []
  for char_dir in train_image_list:
    for file_name in os.listdir(char_dir):
      file_name = char_dir + "/" + file_name
      picked_image = io.imread(file_name)
      image_resize = transform.resize(picked_image, size)
      raw_train_images.append(image_resize)
      raw_train_labels.append(char_dir)

  train_images = []
  train_labels = []
  list_range = np.arange(len(raw_train_images))
  np.random.shuffle(list_range)
  for i in list_range:
    train_images.append(raw_train_images[i])
    train_labels.append(raw_train_labels[i])

  raw_test_images = []
  raw_test_labels = []
  for char_dir in test_image_list:
    for file_name in os.listdir(char_dir):
      file_name = char_dir + "/" + file_name
      picked_image = io.imread(file_name)
      image_resize = transform.resize(picked_image, size)
      raw_test_images.append(image_resize)
      raw_test_labels.append(char_dir)
  
  test_images = []
  test_labels = []
  list_range = np.arange(len(raw_test_images))
  np.random.shuffle(list_range)
  for i in list_range:
    test_images.append(raw_test_images[i])
    test_labels.append(raw_test_labels[i])

  return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = get_images()

#Training information
batchS = 32
nPlanes = 3500

nClasses = 3
nSuppImgs = 5
nSupportTraining = 10000
nTrials = 1000
nSupp = nClasses * nSuppImgs
unseen = False

opts, args = getopt.getopt(sys.argv[1:], "hc:i:s:p:u", ["help", 
  "num_classes=", "num_supports=", "num_iterations=","num_planes=",
  "unseen"])

for o, a in opts:
  if o in ("-c", "--num_classes"):
    nClasses = int(a)
  elif o in ("-s", "--num_supports"):
    nImgsSuppClass = int(a)
  elif o in ("-p", "--num_planes"):
    nPlanes = int(a)
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

nSupp = nClasses * nSuppImgs

tf.reset_default_graph()

# Query Information - Vector of images
dataset = tf.placeholder(tf.float32, [None]+size)

# Network Function, call for each image to create feature vector
def create_network(img, size, First = False):
  currInp = img
  layer = 0
  currFilt = size[2]
  
  for k in nKernels:
    with tf.variable_scope('conv'+str(layer),reuse = 
      tf.AUTO_REUSE) as varscope:
      layer += 1
      weight = tf.get_variable('weight', [3,3,currFilt,k])
      currFilt = k
      bias = tf.get_variable('bias', [k], initializer = 
        tf.constant_initializer(0.0))
      convR = tf.nn.conv2d(currInp, weight, strides=[1,1,1,1], 
        padding="SAME")
      convR = tf.add(convR, bias)
      reluR = tf.nn.relu(convR)
      poolR = tf.nn.max_pool(reluR, ksize=[1,2,2,1], strides=[1,2,2,1], 
        padding="SAME")
      currInp = poolR
  
  return currInp

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
  return np.transpose((np.matlib.rand(np.asarray(feature_vectors).shape[-1], 
    num_planes) - 0.5) * 2), np.zeros(num_planes)
  
    # lsh_offset_vals.append(np.dot(clf.intercept_[0], clf.coef_[0]))
  return lsh_matrix, lsh_offset_vals, clfs

# Calculate Hash for the feature vector
def lsh_hash(feature_vectors, LSH_matrix, lsh_offset_vals):
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
      featureVectors = np.empty([len(rawDataset), FEAT.shape[2],
        FEAT.shape[3], FEAT.shape[4]])
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
      queryFeatureVectors = np.empty([len(queryDataset), FEAT.shape[2],
        FEAT.shape[3], FEAT.shape[4]])
    queryFeatureVectors[i*1000:(i+1)*1000] = FEAT[0]
  queryFeatureVectors = np.reshape(queryFeatureVectors, (len(queryDataset), 
    -1))

sumEff = 0

cos_acc = 0
lsh_acc = 0
lsh_acc2 = 0

# choose random support vectors
if unseen:
  sourceVectors = queryFeatureVectors
  sourceLabels = queryLabels
else:
  sourceVectors = featureVectors
  sourceLabels = rawLabels

supp = []
supp_labels = []
supp_indices = []
while len(supp) < nClasses * nSuppImgs:
  supp_index = random.randint(0, len(sourceLabels) - 1)
  choice = sourceLabels[supp_index]
  while choice in supp_labels:
    supp_index = random.randint(0, len(sourceLabels) - 1)
    choice = sourceLabels[supp_index]
  n = 0
  change = 1
  while n < nSuppImgs:
    while sourceLabels[supp_index] != choice:
      supp_index += 1
      if supp_index == len(sourceLabels):
        supp_index = 0
    n += 1
    supp.append(sourceVectors[supp_index])
    supp_labels.append(sourceLabels[supp_index])
    supp_indices.append(supp_index)

lsh_planes, lsh_offset_vals = gen_lsh_pick_planes(nPlanes, 
  supp, supp_labels)

for i in range(nTrials):
  # choose random query
  query_value = random.choice(supp_labels)
  query_index = random.randint(0, len(sourceLabels) - 1)
  while query_value != sourceLabels[query_index] or query_index in supp_indices:
    query_index += 1
    if query_index == len(sourceLabels):
      query_index = 0
  query = sourceVectors[query_index]
  query_label = sourceLabels[query_index]
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
