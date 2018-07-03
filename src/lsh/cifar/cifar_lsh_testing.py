# LSH Testing for CIFAR

from sklearn import svm
import tensorflow as tf
import numpy as np
import scipy.misc
import pickle
import getopt
import random
import math
import sys
import os

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
nKernels = [64, 64, 64]
fully_connected_nodes = 128
poolS = 2

batchS = 32
nPlanes_list = [500] 
classList = [1,2,3,4,5,6,7,8,9,0]
nClasses_list = [3]
nSuppImgs_list = [5]
nSupportTraining = 10000
nTrials = 1000

hashing_methods=["random", "one_rest"]
unseen_list = [False]
model_dir = None

opts, args = getopt.getopt(sys.argv[1:], "hc:i:s:p:a:u:d:m:", ["help", 
  "num_classes_list=", "num_supports_list=", "num_iterations=",
  "num_planes_list=","unseen_list=","model_dir=", "hashing_methods="])

for o, a in opts:
  if o in ("-c", "--num_classes"):
    nClasses_list = [int(i) for i in a.split(",")]
  elif o in ("-s", "--num_supports"):
    nSuppImgs_list = [int(i) for i in a.split(",")]
  elif o in ("-p", "--num_planes"):
    nPlanes_list = [int(i) for i in a.split(",")]
  elif o in ("-i", "--num_iterations"):
    nTrials = int(a)
  elif o in ("-h", "--help"):
    help_message()
  elif o in ("-u", "--unseen"):
    unseen_list = [True for i in a.split(",") if i == "True"]
  elif o in ("-d", "--model_dir"):
    model_dir = a
  elif o in ("-m", "--hashing_methods"):
    hashing_methods = [i for i in a.split(",") if (i == "one_rest" or 
      i == "random")]
  else:
    print("unhandled option "+o)
    help_message()

if not model_dir:
  print("no list of models provided")
  exit(1)
train_images = []
train_labels = []
list_range = np.arange(len(train_images_raw))
np.random.shuffle(list_range)
for index, i in enumerate(list_range):
  train_images.append(train_images_raw[i])
  train_labels.append(train_labels_raw[i])

train_images = np.reshape(train_images, [len(train_images), 3, 32, 32])
train_images = np.transpose(train_images, [0, 2, 3, 1])

test_images = test[b"data"]
test_images = np.reshape(test_images, [len(test_images), 3, 32, 32])
test_images = np.transpose(test_images, [0, 2, 3, 1])
test_labels = test[b"labels"]

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

# Cosine Similarity 
def cos_similarities(supports, query): 
  dotProduct = np.sum(np.multiply(supports, query), (1))
  supportsMagn = np.sqrt(np.sum(np.square(supports), (1)))
  cosDist = dotProduct / np.clip(supportsMagn, 1e-10, float("inf"))
  return cosDist

def gen_lsh_random_planes(num_planes, feature_vectors, labels):
  return np.transpose((np.matlib.rand(feature_vectors.shape[-1], num_planes) 
    - 0.5) * 2), np.zeros(num_planes)
  
    # lsh_offset_vals.append(np.dot(clf.intercept_[0], clf.coef_[0]))
  return lsh_matrix, lsh_offset_vals, clfs

def gen_lsh_pick_planes(num_planes, feature_vectors, labels, numbers):
  
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

# Calculate Hash for the feature vector
def lsh_hash(feature_vectors, LSH_matrix, lsh_offset_vals):
  lsh_vectors = np.matmul(feature_vectors, np.transpose(LSH_matrix))
  lsh_vectors = np.subtract(lsh_vectors, lsh_offset_vals)
  lsh_bin = np.sign(lsh_vectors)
  lsh_bin = np.clip(lsh_bin, 0, 1)
  return lsh_bin, lsh_vectors

# Generate Distance
def lsh_dist(lshSupp, lshQueryO, lshVecSupp, lshVecQuery, nPlanes):
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

file_objs = {}
for model_style in ["cosine", "lsh_random", "lsh_one_rest"]:
  for method in hashing_methods:
    data_file_name = "../../../data/csv/cifar_"+model_style+"_lsh_"+method+".csv"
    file_objs[data_file_name] = open(data_file_name, 'w')
    first_line = "method,model_classes,model_supports,"
    if model_style == "one_rest":
      first_line += "period,"
    elif model_style == "random":
      first_line += "model_planes,trained_planes,"
    first_line += "testing_classes,testing_supports,"
    if method == "random":
      first_line += "testing_planes,"
    first_line += "unseen,cos_acc,true_lsh_acc,sigmoid_lsh_acc"
    file_objs[data_file_name].write(first_line + "\n")

models_done = set()
for category in os.listdir(model_dir):
  for file_name in os.listdir(model_dir + "/" + category):
    if "cosine" not in file_name and "lsh" not in file_name:
      continue

    model_name = file_name.split(".")[0]
    
    if model_name in models_done:
      continue

    models_done.add(model_name)
    
    SAVE_PATH = model_dir + "/" + category + "/" + model_name

    end_file = file_name.split("-")

    index = 0
    reference_dict = None
    model_style = None
    while not reference_dict:
      if end_file[index] == "cosine":
        model_style = "cosine"
        reference_dict = (("classes", end_file[index + 1]),
                          ("supports", end_file[index+2]))
      elif end_file[index] == "lsh":
        if end_file[index + 1] == "one":
          model_style = "lsh_one_rest"
          reference_dict = (("classes", end_file[index + 4]),
                            ("supports", end_file[index + 5]),
                            ("period", end_file[index + 3]))
        else:
          model_style = "lsh_random"
          reference_dict = (("classes", end_file[index + 3]),
                            ("supports", end_file[index + 4]),
                            ("planes", end_file[index + 2]),
                            ("training", end_file[index + 5]))
      index+=1
    
    for method in hashing_methods:
      for nPlanes in nPlanes_list:
        for nClasses in nClasses_list:
          if method == "one_rest":
            nPlanes = nClasses
          for unseen in unseen_list:
              if unseen:
                numbers = classList[10-nClasses:]
              else:
                numbers = classList[:nClasses]
              for nSuppImgs in nSuppImgs_list:
                tf.reset_default_graph()

                nSupp = nClasses * nSuppImgs

                # Query Information - Vector of images
                dataset = tf.placeholder(tf.float32, [None]+size)

                features = create_network(dataset, size)

                init = tf.global_variables_initializer()

                with tf.Session() as session:
                  session.run(init)

                  Saver = tf.train.Saver()
                  Saver.restore(session, SAVE_PATH)
                 
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
                      queryFeatureVectors = np.empty([len(queryDataset), 
                        FEAT.shape[2], FEAT.shape[3], FEAT.shape[4]])
                    queryFeatureVectors[i*1000:(i+1)*1000] = FEAT[0]
                  queryFeatureVectors = np.reshape(queryFeatureVectors, (len(queryDataset), -1))

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
                    supp_index = random.randint(0, train_images.shape[0] - 1)
                    while int(train_labels[supp_index]) != j:
                      supp_index += 1
                      if supp_index == len(train_images):
                        supp_index = 0
                    n += 1
                    supp.append(featureVectors[supp_index])
                    supp_labels.append(int(train_labels[supp_index]))

                if method == "random":
                  lsh_planes, lsh_offset_vals = gen_lsh_random_planes(nPlanes, featureVectors[:nSupportTraining], rawLabels)
                elif method == "one_rest":
                  lsh_planes, lsh_offset_vals = gen_lsh_pick_planes(nPlanes, supp, supp_labels, numbers)

                for i in range(nTrials):
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
                  lsh_bin_q, lsh_vec_q = lsh_hash(np.asarray(query), lsh_planes, lsh_offset_vals)

                  # calculate lsh distances
                  # find closest match
                  distances, distances2 = lsh_dist(lsh_bin, lsh_bin_q, lsh_vec, lsh_vec_q, nPlanes)
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

                cos_lsh_acc = float(cos_acc)/(nTrials)
                calc_lsh_acc = float(lsh_acc)/(nTrials)
                calc_lsh_acc2 = float(lsh_acc2)/(nTrials)
                eff = float(sumEff) / nTrials
                output_file = "../../../data/csv/cifar_"+model_style+"_lsh_"+method+".csv"
                output="lsh_"+method+","
                for i in reference_dict:
                  output += i[1] + ","
                output += str(nClasses)+","+str(nSuppImgs)+","
                if method == "random":
                  output+=str(nPlanes)+","
                output += str(unseen)+","+str(cos_lsh_acc) + "," + str(calc_lsh_acc) + "," + str(calc_lsh_acc2)
                print(output_file)
                print(output)
                file_objs[output_file].write(output + "\n")
        if method == "one_rest":
          break

for file_obj in file_obs.keys():
  file_objs[file_obj].close()  
