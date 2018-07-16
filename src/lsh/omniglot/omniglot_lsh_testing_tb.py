# Checking viability of LSH with random planes Omniglot

from skimage import transform, io
from sklearn import svm
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
# LSH Testing
batchS = 32
nPlanes_list = [500] 

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

def cos_similarities(supports, query): 
  dotProduct = tf.reduce_sum(tf.multiply(supports, query), (1))
  supportsMagn = tf.sqrt(tf.reduce_sum(tf.square(supports), (1)))
  cosDist = tf.divide(dotProduct, tf.clip_by_value(supportsMagn, 1e-10, float("inf")))
  return cosDist

def gen_lsh_pick_planes(num_planes, feature_vectors, labels):
  
  lsh_matrix = []
  lsh_offset_vals = []
  
  feature_vectors = np.reshape(np.asarray(feature_vectors),
    (len(feature_vectors), -1))

  label_dirs = set(labels)
  for index_i, i in enumerate(label_dirs):
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
    lsh_offset_vals.append(clf.intercept_[0])

  return lsh_matrix, lsh_offset_vals

def gen_lsh_random_planes(num_planes, feature_vectors, labels):
  return np.transpose((np.matlib.rand(feature_vectors.shape[-1], num_planes) - 0.5) * 2), np.zeros(num_planes)

def lsh_true_hash(feature_vectors, LSH_matrix, lsh_offset_vals):
  with tf.name_scope("true_lsh_hashing"):
    lsh_vectors = tf.matmul(feature_vectors, tf.transpose(LSH_matrix))
    lsh_vectors = tf.add(lsh_vectors, lsh_offset_vals)
    lsh_bin = tf.sign(lsh_vectors)
    lsh_bin = tf.clip_by_value(lsh_bin, 0, 1)
    return lsh_bin

def lsh_sig_hash(feature_vectors, LSH_matrix, lsh_offset_vals):
  with tf.name_scope("sigmoid_lsh_hashing"):
    lsh_vectors = tf.matmul(feature_vectors, tf.transpose(LSH_matrix))
    lsh_vectors = tf.add(lsh_vectors, lsh_offset_vals)
    return lsh_vectors

  # Generate distance
def true_lsh_dist(lshSupp, lshQueryO):
  with tf.name_scope("true_lsh_distance"):
    lshQuery = []
    for i in range(nSupp):
      lshQuery.append(lshQueryO)
    lshQuery = tf.stack(lshQuery)
    lshQuery = tf.reshape(lshQuery, [lshSupp.shape[0], lshSupp.shape[1]])
    dist = tf.equal(lshSupp, lshQuery)
    dist = tf.reduce_sum(tf.cast(dist, tf.int32), [1])
    return dist

def sigmoid_lsh_dist(lshVecSupp, lshVecQuery):
  with tf.name_scope("sigmoid_lsh_distance"):
    lshQuery = []
    for i in range(nSupp):
      lshQuery.append(lshVecQuery)
    lshQuery = tf.reshape(tf.stack(lshQuery), [lshVecSupp.shape[0], lshVecSupp.shape[1]])
    dist_2 = tf.multiply(lshVecSupp, lshQuery)
    dist2 = tf.divide(1.0, np.add(1.0, tf.exp(tf.multiply(-50.0, dist_2))))
    dist2 = tf.reduce_sum(dist2, [1])  # check this!
    return dist2

file_objs = {}
for model_style in ["cosine"]:#, "lsh_random", "lsh_one_rest"]:
  for method in hashing_methods:
    data_file_name = "../../../data/csv/omniglot_normalization_"+model_style+"_lsh_"+method+".csv"
    file_objs[data_file_name] = open(data_file_name, 'w')
    first_line = "method,model_classes,model_supports,"
    if model_style == "one_rest":
      first_line += "model_period,"
    elif model_style == "random":
      first_line += "model_planes,model_trained_planes,"
    first_line += "testing_classes,testing_supports,"
    if method == "random":
      first_line += "testing_planes,"
    first_line += "unseen,cos_acc,true_lsh_acc,sigmoid_lsh_acc"
    file_objs[data_file_name].write(first_line + "\n")

models_done = set()
for category in os.listdir(model_dir):
  if category[0] == ".":
    continue
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
          reference_dict = (("classes", end_file[index + 4]),
                            ("supports", end_file[index + 5]),
                            ("planes", end_file[index + 3]),
                            ("training", end_file[index + 2]))
      index+=1
    
    tf.reset_default_graph()

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
      for i in range(int(len(rawDataset)/batchS)):
        FEAT = (session.run([features], feed_dict =
          {dataset: rawDataset[i*batchS:(i+1)*batchS]}))
        FEAT = np.asarray(FEAT)
        if featureVectors is None:
          featureVectors = np.empty([len(rawDataset), FEAT.shape[2], 
            FEAT.shape[3], FEAT.shape[4]])
        featureVectors[i*batchS:(i+1)*batchS] = FEAT[0]
      featureVectors = np.reshape(featureVectors, (len(rawDataset), -1))  

      queryDataset = test_images
      queryLabels = test_labels

      queryFeatureVectors = None
      for i in range(int(len(queryDataset)/batchS)):
        FEAT = (session.run([features], feed_dict =
          {dataset: queryDataset[i*batchS:(i+1)*batchS]}))
        FEAT = np.asarray(FEAT)
        if queryFeatureVectors is None:
          queryFeatureVectors = np.empty([len(queryDataset), 
            FEAT.shape[2], FEAT.shape[3], FEAT.shape[4]])
        queryFeatureVectors[i*batchS:(i+1)*batchS] = FEAT[0]
      queryFeatureVectors = np.reshape(queryFeatureVectors, (len(queryDataset), -1))

    for method in hashing_methods:
      for nPlanes in nPlanes_list:
        for nClasses in nClasses_list:
          if method == "one_rest":
            nPlanes = nClasses
          for unseen in unseen_list:
              for nSuppImgs in nSuppImgs_list:
                tf.reset_default_graph()

                nSupp = nClasses * nSuppImgs

                support_vectors = tf.placeholder(tf.float32, [nSupp, featureVectors.shape[1]], name = "support_feature_vectors")
                query_vector = tf.placeholder(tf.float32, [1, queryFeatureVectors.shape[1]], name = "query_feature_vector")
                lsh_planes_tf = tf.placeholder(tf.float32, [nPlanes, queryFeatureVectors.shape[1]], name = "lsh_planes")
                lsh_offsets_tf = tf.placeholder(tf.float32, [nPlanes], name = "lsh_offsets")

                with tf.name_scope("cosine"):
                  cosine_distances = cos_similarities(support_vectors, query_vector)
                with tf.name_scope("true_lsh"):
                  supp_true = lsh_true_hash(support_vectors, lsh_planes_tf, lsh_offsets_tf)
                  query_true = lsh_true_hash(query_vector, lsh_planes_tf, lsh_offsets_tf)
                  true_lsh_distances = true_lsh_dist(supp_true, query_true)
                with tf.name_scope("sigmoid_lsh"):
                  supp_sig = lsh_sig_hash(support_vectors, lsh_planes_tf, lsh_offsets_tf)
                  query_sig = lsh_sig_hash(query_vector, lsh_planes_tf, lsh_offsets_tf)
                  sigmoid_lsh_distances = sigmoid_lsh_dist(supp_sig, query_sig)
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

                if method == "random":
                  lsh_planes, lsh_offset_vals = gen_lsh_random_planes(nPlanes, featureVectors[:nSupportTraining], rawLabels)
                elif method == "one_rest":
                  lsh_planes, lsh_offset_vals = gen_lsh_pick_planes(nPlanes, supp, supp_labels)

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
                
                  with tf.Session() as session:
                    cosDistances, distancesTrue, distancesSig = session.run(
                      [cosine_distances, true_lsh_distances,
                      sigmoid_lsh_distances], feed_dict = {
                      query_vector: [query],
                      support_vectors: supp,
                      lsh_planes_tf: lsh_planes,
                      lsh_offsets_tf: lsh_offset_vals
                      })

                  LSHMatchTrue = supp_labels[np.argmax(distancesTrue)]
                  LSHMatchSig = supp_labels[np.argmax(distancesSig)]
                  cosMatch = supp_labels[np.argmax(cosDistances)]
               
                  if cosMatch == query_label:
                    cos_acc += 1
                
                  if LSHMatchTrue == query_label:
                    lsh_acc+=1

                  if LSHMatchSig == query_label:
                    lsh_acc2+=1   

                cos_lsh_acc = float(cos_acc)/(nTrials)
                calc_lsh_acc = float(lsh_acc)/(nTrials)
                calc_lsh_acc2 = float(lsh_acc2)/(nTrials)
                output_file = "../../../data/csv/omniglot_normalization_"+model_style+"_lsh_"+method+".csv"
                output="lsh_"+method+","
                for i in reference_dict:
                  output += i[1] + ","
                output += str(nClasses)+","+str(nSuppImgs)+","
                if method == "random":
                  output+=str(nPlanes)+","
                output += str(unseen)+","+str(cos_lsh_acc) + "," + str(calc_lsh_acc) + "," + str(calc_lsh_acc2)
                print(output)
                #file_objs[output_file].write(output + "\n")
        if method == "one_rest":
          break

for file_obj in file_objs.keys():
  file_objs[file_obj].close() 
