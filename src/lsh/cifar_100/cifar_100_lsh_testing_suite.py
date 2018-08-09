# Checking viability of LSH with random planes Omniglot

import os

#os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"] + ":/afs/crc.nd.edu/x86_64_linux/c/cuda/8.0/extras/CUPTI/lib64/"

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from skimage import transform, io
from sklearn import svm
import tensorflow as tf
import numpy as np
import scipy.misc
import getopt
import random
import math
import sys
import pickle

LOG_DIR = "./cifar_100_testing"

train_file_path = "../../../testing-data/cifar-100/train"
train_images_raw = np.empty((0, 3072))
train_labels_raw = np.empty((0))
with open(train_file_path, 'rb') as cifar_file:
  data = pickle.load(cifar_file)
  train_images_raw = data[b"data"]
  train_labels_raw = data[b"fine_labels"]

test_file_name = "../../../testing-data/cifar-100/test"
with open(test_file_name, 'rb') as cifar_file:
  test = pickle.load(cifar_file)

# Graph Constants
size = [32, 32, 3]
nKernels = [64, 64, 64]
fully_connected_nodes = 128
poolS = 2

# LSH Testing
batchS = 32
nPlanes_list = [500] 

nClasses_list = [3]
nSuppImgs_list = [5]
nSupportTraining = 10000
nTrials = 1000

hashing_methods=["random", "one_rest"]
unseen_list = {True}
model_dir = None
one_model = False
file_write = False
file_write_path = None
tensorboard = False
integer_range_list = [1000]
integer_change = False
new_set = False
batch_norm = False
verbose = True
dropout = False

def help_message():
  print("Testing Suite LSH Hashing for variable number of items\n")

  print("lists are comma separated values")

  print("Options:")
  print("-f,--file_write:         write info to file")
  print("-b,--batch_norm:         use batch norm model")
  print("-t,--tensorboard:        write info to file")
  print("-c,--num_classes_list:   input list of classes")
  print("-r,--integer_range:      list of ranges for integers for plane")
  print("-s,--num_supports_list:  input list of supports")
  print("-p,--num_planes_list:    input list of number of planes")
  print("-u,--unseen_list:        list of True False to for unseen data")
  print("-d,--model_dir:          directory with list of models")
  print("-l,--model_loc:          path to model location")
  print("-m,--hashing_methods:    list of hashing methods to use")
  print("-i,--num_iterations:     number of test iterations")
  print("-n,--new_support_set:    get new support set for each trial")
  print("-v,--non_verbose:        no results printed to console")
  print("")
  exit(1)

opts, args = getopt.getopt(sys.argv[1:], "hnbvtf:c:r:s:p:u:d:i:m:l:", ["help", 
  "batch_norm","new_support_set","num_classes_list=", "num_supports_list=", 
  "num_iterations=","num_planes_list=","unseen_list=","model_dir=",
  "hashing_methods=","model_loc=","integer_range=","tensorboard=",
  "file_write=","no_verbose","tensorboard"])

for o, a in opts:
  if o in ("-c", "--num_classes"):
    nClasses_list = [int(i) for i in a.split(",")]
  elif o in ("-f", "--file_write"):
    file_write = True
    if a and a[0] != "-":
      file_write_path = a
  elif o in ("--integer_range", "-r"):
    integer_range_list = [int(i) for i in a.split(",")]
    integer_change = True
  elif o in ("-s", "--num_supports"):
    nSuppImgs_list = [int(i) for i in a.split(",")]
  elif o in ("-p", "--num_planes"):
    nPlanes_list = [int(i) for i in a.split(",")]
  elif o in ("-i", "--num_iterations"):
    nTrials = int(a)
  elif o in ("-h", "--help"):
    help_message()
  elif o in ("-u", "--unseen"):
    [unseen_list.add(True) if i == "True" else unseen_list.add(False) for i in a.split(",")]
  elif o in ("-d", "--model_dir"):
    if one_model:
      print("using predetermined model location")
      continue
    model_dir = a
  elif o in ("-t", "--tensorboard"):
    tensorboard = True
    if a and a[0] != "-":
      LOG_DIR = a
  elif o in ("-l", "--model_loc"):
    if model_dir is not None:
      print("using predetermined model list")
      continue
    model_dir = a
    one_model = True
  elif o in ("-m", "--hashing_methods"):
    hashing_methods = [i for i in a.split(",") if (i == "one_rest" or 
      i == "random")]
  elif o in ("-b", "--batch_norm"):
    batch_norm = True
  elif o in ("-n", "--new_support_set"):
    new_set = True
  elif o in ("-v", "--non_verbose"):
    verbose = False
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
      convR = tf.nn.conv2d(currInp, weight, strides=[1,1,1,1],
        padding= "SAME")
      if batch_norm:
        beta = tf.get_variable('beta', [k], initializer = tf.constant_initializer(0.0))     
        gamma = tf.get_variable('gamma', [k], initializer=tf.constant_initializer(1.0))     
        mean, variance = tf.nn.moments(convR, [0,1,2])
        PostNormalized = tf.nn.batch_normalization(convR,mean,variance,beta,gamma,1e-10)
        reluR = tf.nn.relu(PostNormalized)
      else:
        bias = tf.get_variable('bias', [k], initializer = 
          tf.constant_initializer(0.0))
        convR = tf.add(convR, bias)
        reluR = tf.nn.relu(convR)
      poolR = tf.nn.max_pool(reluR, ksize=[1,2,2,1], strides=[1,2,2,1], 
        padding="SAME")
      currInp = poolR
  
  if dropout:
    currInp = tf.nn.dropout(currInp, 0.8)

  return currInp

def cos_similarities(supports, query):
  with tf.name_scope("cosine_distance"):
    dotProduct = tf.reduce_sum(tf.multiply(supports, query, name = "feature_multiply"), (1), name = "product_add")
    supportsMagn = tf.sqrt(tf.reduce_sum(tf.square(supports, name = "mag_sqaure"), (1)), name = "mag_sqrt")
    cosDist = tf.divide(dotProduct, tf.clip_by_value(supportsMagn, 1e-10, float("inf")), name = "cos_div")
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
    
    lsh_offset_vals.append(clf.intercept_[0])

  return lsh_matrix, lsh_offset_vals

def gen_lsh_random_planes(num_planes, feature_vectors, labels):
  return ((np.matlib.rand(feature_vectors.shape[-1], num_planes) - 0.5) * 2), np.zeros(num_planes)

def gen_lsh_random_int_planes(num_planes, feature_vectors, labels):
  return np.random.randint(-integer_range, integer_range, size = (feature_vectors.shape[-1], num_planes)), np.zeros(num_planes)

def lsh_true_hash(lsh_bin):
  with tf.name_scope("true_lsh_hashing"):
    lsh_bin = tf.sign(lsh_bin, name = "bin_signs")
    lsh_bin = tf.clip_by_value(lsh_bin, 0, 1, name = "clip")
    lsh_bin = tf.cast(lsh_bin, bool)
    return lsh_bin

def lsh_sig_hash(feature_vectors, LSH_matrix, lsh_offset_vals):
  with tf.name_scope("sigmoid_lsh_hashing"):
    lsh_vectors = tf.matmul(feature_vectors, LSH_matrix, name = "matrix_multiply")
    lsh_vectors = tf.add(lsh_vectors, lsh_offset_vals, name = "offset_addition")
    return lsh_vectors

  # Generate distance
def true_lsh_dist(lshSupp, lshQueryO):
  with tf.name_scope("true_lsh_distance"):
    dist = tf.logical_not(tf.logical_xor(lshSupp, lshQueryO))
    dist = tf.reduce_sum(tf.cast(dist, tf.int32), [1])
    return dist

def true_lsh_dist_equal(lshSupp, lsh_bin):
  with tf.name_scope("true_lsh_hashing"):
    lsh_bin = tf.sign(lsh_bin, name = "bin_signs")
    lsh_bin = tf.clip_by_value(lsh_bin, 0, 1, name = "clip")
    lsh_bin = tf.cast(lsh_bin, bool)
  with tf.name_scope("true_lsh_distance"):
    dist = tf.equal(lshSupp, lsh_bin)
    dist = tf.reduce_sum(tf.cast(dist, tf.int32), [1])
    return dist

def sigmoid_lsh_dist(lshVecSupp, lshVecQuery):
  with tf.name_scope("sigmoid_lsh_distance"):
    dist_2 = tf.multiply(lshVecSupp, lshVecQuery)
    dist2 = tf.divide(1.0, np.add(1.0, tf.exp(tf.multiply(-1.0, dist_2))))
    dist2 = tf.reduce_sum(dist2, [1])  # check this!
    return dist2

file_objs = {}
for model_style in ["cosine", "lsh_random", "lsh_one_rest"]:
  for method in hashing_methods:
    data_file_name = "cifar_100_"
    data_file_name += model_style+"_lsh_"+method+".csv"
    if file_write and file_write_path:
      file_objs[data_file_name] = open(file_write_path + "/" + data_file_name, 'w')
    elif file_write:
      file_objs[data_file_name] = open(base_path +"/data/csv/"+data_file_name, 'w')
    first_line = "method,model_batch_norm,model_dropout,model_layers,model_classes,model_supports,"
    if model_style == "lsh_one_rest":
      first_line += "model_period,"
    elif model_style == "lsh_random":
      first_line += "model_planes,model_trained_planes,"
    first_line += "testing_classes,testing_supports,"
    if method == "random":
      first_line += "testing_planes,"
    first_line += "integer_range,unseen,cos_acc,true_lsh_acc,sigmoid_lsh_acc"
    if file_write:
      file_objs[data_file_name].write(first_line + "\n")

if one_model:
  model_list = ["one"]
else:
  model_list = os.listdir(model_dir)
models_done = set()
for category in model_list:
  if one_model:
    categories = [""]
  else:
    categories = os.listdir(model_dir + "/" + category)
  for file_name in categories:
    if one_model:
      file_name = model_dir.split("/")[-1]
    if "cosine" not in file_name and "lsh" not in file_name:
      continue

    model_name = file_name.split(".")[0]

    if model_name in models_done:
      continue

    models_done.add(model_name)
    if one_model:
      SAVE_PATH = model_dir
    else:
      SAVE_PATH = model_dir + "/" + category + "/" + model_name    

    end_file = file_name.split("-")

    index = 0
    reference_dict = None
    model_style = None
    while not reference_dict:
      if end_file[index] == "cosine":
        model_style = "cosine"
        if end_file[index + 1] == "norm":
          index += 1
          batch_norm = True
        if end_file[index + 1] == "dropout":
          index += 1
          dropout = True
        #nKernels = [64 for x in range(int(end_file[index+1]))]
        reference_dict = (#("nLayers", end_file[index + 1]),
                          ("classes", end_file[index + 2-1]),
                          ("supports", end_file[index+3-1]))
      elif end_file[index] == "lsh":
        if end_file[index + 1] == "one":
          index += 2
          model_style = "lsh_one_rest"
          if end_file[index + 1] == "norm":
            index += 1
            batch_norm = True
          if end_file[index + 1] == "dropout":
            index += 1
            dropout = True
          #nKernels = [64 for x in range(int(end_file[index+1]))]
          reference_dict = (#("nLayers", end_file[index + 1]),
                            ("classes", end_file[index + 3-1]),
                            ("supports", end_file[index + 4-1]),
                            ("period", end_file[index + 2-1]))
        else:
          model_style = "lsh_random"
          index += 1
          if end_file[index + 1] == "norm":
            index += 1
            batch_norm = True
          if end_file[index + 1] == "dropout":
            index += 1
            dropout = True
          #nKernels = [64 for x in range(int(end_file[index+1]))]
          reference_dict = (#("nLayers", end_file[index + 1]),
                            ("classes", end_file[index + 4 - 1]),
                            ("supports", end_file[index + 5- 1]),
                            ("planes", end_file[index + 3-1]),
                            ("training", end_file[index + 2-1]))
      index+=1
    
    tf.reset_default_graph()

    # Query Information - Vector of images
    dataset = tf.placeholder(tf.float32, [None]+size)

    features = create_network(dataset, size)

    init = tf.global_variables_initializer()
    for unseen in unseen_list:
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

      for method in hashing_methods:
        for nPlanes in nPlanes_list:
          for nClasses in nClasses_list:
            if method == "one_rest":
              nPlanes = nClasses
            for nSuppImgs in nSuppImgs_list:
              for integer_range in integer_range_list:  
                tf.reset_default_graph()

                nSupp = nClasses * nSuppImgs
                with tf.name_scope("support_setup"):
                  support_vectors = tf.placeholder(tf.float32, [nSupp, featureVectors.shape[1]], name = "support_feature_vectors")
                  lsh_planes_tf = tf.placeholder(tf.float32, [queryFeatureVectors.shape[1], nPlanes], name = "lsh_planes")
                  lsh_offsets_tf = tf.placeholder(tf.float32, [nPlanes], name = "lsh_offsets")

                  supp_sig = lsh_sig_hash(support_vectors, lsh_planes_tf, lsh_offsets_tf)
                  supp_true = lsh_true_hash(supp_sig)

                query_vector = tf.placeholder(tf.float32, [1, queryFeatureVectors.shape[1]], name = "query_feature_vector")
                summary = tf.summary.merge_all()
                #with tf.name_scope("cosine"):
                #  with tf.name_scope("cosine_distance"):
                with tf.name_scope("cosine_distance"):
                  dotProduct = tf.reduce_sum(tf.multiply(support_vectors, query_vector, name = "feature_multiply"), (1), name = "product_add")
                  supportsMagn = tf.sqrt(tf.reduce_sum(tf.square(support_vectors, name = "mag_sqaure"), (1)), name = "mag_sqrt")
                  cosine_distances = tf.divide(dotProduct, tf.clip_by_value(supportsMagn, 1e-10, float("inf")), name = "cos_div")
                  
                #with tf.name_scope("true_lsh"):
                #  query_true = lsh_true_hash(query_vector, lsh_planes_tf, lsh_offsets_tf, "")
                #  true_lsh_distances = true_lsh_dist(supp_true, query_true)
                #with tf.name_scope("first_hash"):
                lsh_vector = lsh_sig_hash(query_vector, lsh_planes_tf, lsh_offsets_tf)
                #with tf.name_scope("true_lsh_equal"):
                  #with tf.name_scope("true_lsh_hashing"):
                with tf.name_scope("true_lsh_hashing"):
                  lsh_bin = tf.sign(lsh_vector, name = "bin_signs")
                  lsh_bin = tf.clip_by_value(lsh_bin, 0, 1, name = "clip")
                  lsh_bin = tf.cast(lsh_bin, bool)
                  #with tf.name_scope("true_lsh_distance"):
                  dist = tf.equal(supp_true, lsh_bin)
                  true_lsh_distances = tf.reduce_sum(tf.cast(dist, tf.int32), [1])
                #with tf.name_scope("sigmoid_lsh"):
                  #with tf.name_scope("sigmoid_lsh_distance"):
                with tf.name_scope("sig_lsh_hashing"):
                  dist_2 = tf.multiply(supp_sig, lsh_vector)
                  dist2 = tf.divide(1.0, np.add(1.0, tf.exp(tf.multiply(-50.0, dist_2))))
                  sigmoid_lsh_distances = tf.reduce_sum(dist2, [1])  # check this!
                
                sumEff = 0
                cos_acc = 0
                lsh_acc = 0
                lsh_acc2 = 0
                # choose random support vectors

                if method == "random":
                  lsh_planes, lsh_offset_vals = gen_lsh_random_int_planes(nPlanes, featureVectors[:nSupportTraining], rawLabels)

                if not new_set:
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

                  if method == "one_rest":
                    lsh_planes, lsh_offset_vals = gen_lsh_pick_planes(nPlanes, supp, supp_labels)
                    lsh_planes = np.transpose(lsh_planes)

                with tf.Session() as session:
                  session.run(tf.global_variables_initializer())

                  for i in range(nTrials):
                  
                    if new_set:
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

                      if method == "one_rest":
                        lsh_planes, lsh_offset_vals = gen_lsh_pick_planes(nPlanes, supp, supp_labels)
                        lsh_planes = np.transpose(lsh_planes)
                    # choose random query
                    # choose random query
                    query_value = random.choice(images)
                    query_index = random.randint(0, test_images.shape[0] - 1)
                    while query_value != int(queryLabels[query_index]):
                      query_index += 1
                      if query_index == len(test_images):
                        query_index = 0
                    query = queryFeatureVectors[query_index]
                    query_label = queryLabels[query_index]

                    if i == 1 and tensorboard:
                      batching = "non_batch/"
                      if batch_norm:
                        batching = "batch/"
                      writer = tf.summary.FileWriter(LOG_DIR + "/" + model_style + "/" + batching + method + "/" + str(nPlanes) + "/" + str(nClasses) +"/" + str(nSuppImgs) + "/" + str(i), session.graph)
                      runOptions = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
                      run_metadata = tf.RunMetadata()
                      cosDistances, distancesTrue, distancesSig = session.run(
                      [cosine_distances, true_lsh_distances, #true_lsh_distances_equal,
                      sigmoid_lsh_distances], feed_dict = {
                      query_vector: [query],
                      support_vectors: supp,
                      lsh_planes_tf: lsh_planes,
                      lsh_offsets_tf: lsh_offset_vals
                      }, options = runOptions, run_metadata=run_metadata)
                      writer.add_run_metadata(run_metadata, 'step%d' % i)
                    else:
                      cosDistances, distancesTrue, distancesSig = session.run(
                      [cosine_distances, true_lsh_distances, #true_lsh_distances_equal,
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
                output_file = "omniglot_"
                output_file += model_style+"_lsh_"+method+".csv"
                output="lsh_"+method+","
                output += str(batch_norm)+","+str(dropout) + ","
                for i in reference_dict:
                  output += i[1] + ","
                output += str(nClasses)+","+str(nSuppImgs)+","
                if method == "random":
                  output+=str(nPlanes)+","
                output += str(integer_range)+","
                output += str(unseen)+","+str(cos_lsh_acc) + "," + str(calc_lsh_acc) + "," + str(calc_lsh_acc2)
                if verbose:
                  print(output)
                if file_write:
                  file_objs[output_file].write(output + "\n")
          if method == "one_rest":
            break

for file_obj in file_objs.keys():
  file_objs[file_obj].close() 
