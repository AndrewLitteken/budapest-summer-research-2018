# Using Random LSH Planes training in a matching network CIFAR

import tensorflow as tf
import numpy as np
import getopt
import pickle
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
    data = pickle.load(cifar_file)
    train_images_raw = np.concatenate((train_images_raw, data[b"data"]), 
      axis = 0)
    train_labels_raw = np.concatenate((train_labels_raw, data[b"labels"]), 
      axis = 0)

test_file_name = "../../../testing-data/cifar/test_batch"
with open(test_file_name, 'rb') as cifar_file:
  test = pickle.load(cifar_file)

# Graph Constants
size = [32, 32, 3]
nKernels = [64, 64, 64]
fully_connected_nodes = 128
poolS = 2

#Training information
nIt = 5000
check = 1000
batchS = 32
nPlanes = 100
learning_rate = 1e-10

# Support and testing information
classList = [1,2,3,4,5,6,7,8,9,0]
numbers = []
numbersTest = []
nClasses = 3
batch_norm = False
dropout = False
integer_range = 100

nImgsSuppClass = 5

training = False

base = "/tmp/cifar-lsh-random-"

opts, args = getopt.getopt(sys.argv[1:], "hmnodL:c:i:b:s:", ["help", 
  "num_classes=", "num_supports=", "base_path=", "num_iterations=",
  "dropout", "batch_norm", "num_layers=", "num_planes="])

for o, a in opts:
  if o in ("-t"):
    training = True
  if o in ("-c", "--num_classes"):
    nClasses = int(a)
  elif o in ("-s", "--num_supports"):
    nImgsSuppClass = int(a)
  elif o in ("-b", "--base_path"):
    base = a
    if a[-1] != "/":
      base += "/"
    base += "omniglot-cosine-"
  elif o in ("-i", "--num_iterations"):
    nIt = int(a)
  elif o in ("-d", "--data"):
    train_file_path = "../../../testing-data/omniglot-rotate/"
  elif o in ("-m", "--meta_tensorboard"):
    tensorboard = True
  elif o in ("-o", "--dropout"):
    dropout = True
  elif o in ("-n", "--batch_norm"):
    batch_norm = True
  elif o in ("-p", "--num_planes"):
    nPlanes= int(a)
  elif o in ("-L", "--num_layers"):
    nKernels = [64 for x in range(int(a))]
  elif o in ("-h", "--help"):
    help_message()
  else:
    print("unhandled option: "+o)
    help_message()

numbers = classList[:nClasses]
numbersTest = classList[10-nClasses:]

SAVE_PATH= base + str(len(nKernels)) + "-" + str(nPlanes) + "-" + str(training)+ "-" + str(nClasses) + "-" + str(nImgsSuppClass)

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

# Collecting sample both for query and for testing
def get_samples(class_num, nSupportImgs, testing = False):
  one_hot_list = [0.] * 10
  one_hot_list[class_num] = 1.
  samples = 0
  if not testing:
    imageNum = random.randint(0, len(train_images) - 1)
  else:
    imageNum = random.randint(0, len(test_images) - 1)
  pickedImages = []
  pickedLabels = []
  while samples < nSupportImgs:
    if (imageNum == len(train_images) and not testing):
      imageNum = 0
    elif (imageNum == len(test_images) and testing):
      imageNum = 0
    if not testing:
      labelThis = train_labels[imageNum]
    else:
      labelThis = test_labels[imageNum]
    if labelThis == np.argmax(one_hot_list):
      if not testing:
        imgReshape = np.reshape(train_images[imageNum], [3,32,32])
        imgReshape = np.transpose(imgReshape, [1,2,0])
        pickedLabels.append(train_labels[imageNum])
      else:
        imgReshape = np.reshape(test_images[imageNum], [3,32,32])
        imgReshape = np.transpose(imgReshape, [1,2,0])
        pickedLabels.append(test_labels[imageNum])
      pickedImages.append(imgReshape)
      samples += 1
    imageNum += 1
  return pickedImages, pickedLabels

# Get several images
def get_support(test=False):
  supportImgs = []
  
  choices = numbers
  
  for support in choices:
    newSupportImgs, newSupportLabels = get_samples(support, nImgsSuppClass, 
      test)
    supportImgs.append(newSupportImgs)
  
  return supportImgs

# Get a single query value
def get_query(test=False):
  choices = numbers
  imageInd = random.randint(0, len(choices) - 1)
  imageNum = choices[imageInd]
  img, label = get_samples(imageNum, 1)
  l=np.zeros(len(choices))
  l[imageInd]=1		
  return img[0], l

tf.reset_default_graph()

# Support information - matrix
# Dimensions: batch size, n classes, n supp imgs / class
s_imgs = tf.placeholder(tf.float32, [batchS, nClasses, nImgsSuppClass]+size)

# Query Information - vector
q_img = tf.placeholder(tf.float32, [batchS]+size) # batch size, size
# batch size, number of categories
q_label = tf.placeholder(tf.int32, [batchS, len(numbers)])

# Network Function
# Call for each support image (row of the support matrix) and for the 
# query image.

def create_network(img, size, First = False):
  currInp = img
  layer = 0
  currFilt = size[2]
  
  with tf.name_scope("run_network"):
    for k in nKernels:
      with tf.variable_scope('conv'+str(layer), 
        reuse=tf.AUTO_REUSE) as varscope:
        layer += 1
        weight = tf.get_variable('weight', [3,3,currFilt,k])
        currFilt = k
        if batch_norm:
          convR = tf.nn.conv2d(currInp, weight, strides=[1,1,1,1], padding="SAME")
          beta = tf.get_variable('beta', [k], initializer = tf.constant_initializer(0.0))
          gamma = tf.get_variable('gamma', [k], initializer=tf.constant_initializer(1.0))
          mean, variance = tf.nn.moments(convR, [0,1,2])
          PostNormalized = tf.nn.batch_normalization(convR,mean,variance,beta,gamma,1e-10)
          reluR = tf.nn.relu(PostNormalized)
        else:
          bias = tf.get_variable('bias', [k], initializer = 
            tf.constant_initializer(0.0))
          convR = tf.nn.conv2d(currInp, weight, strides=[1,1,1,1], padding="SAME")
          convR = tf.add(convR, bias)
          reluR = tf.nn.relu(convR)
        poolR = tf.nn.max_pool(reluR, ksize=[1,poolS,poolS,1], 
          strides=[1,poolS,poolS,1], padding="SAME")
        currInp = poolR

    if dropout:
      currInp = tf.nn.dropout(currInp,0.8); 
    return currInp

def generate_lsh_planes(features, nPlanes):
  with tf.variable_scope('lshPlanes', reuse=tf.AUTO_REUSE) as varscope:
    # Generate enough planes of random slopes
    plane = tf.get_variable('plane', initializer = tf.multiply(tf.subtract
      (tf.random_uniform([tf.cast(features.shape[1] * features.shape[2] * 
      features.shape[3], tf.int32), nPlanes], minval = 0, maxval = 1),
      tf.constant(0.5)), tf.constant(2.0)), trainable=training)

    offset = tf.get_variable('offsets', initializer = tf.zeros([nPlanes], 
      tf.float32), trainable = False)

  return plane, offset

def generate_int_lsh_planes(features, nPlanes):
  with tf.variable_scope('lshPlanes', reuse=tf.AUTO_REUSE) as varscope:
    # Generate enough planes of random slopes
    plane = tf.get_variable('plane', initializer = 
      (tf.random_uniform([tf.cast(features.shape[1] * features.shape[2] * 
      features.shape[3], tf.int32), nPlanes], minval = -integer_range, maxval = integer_range)),
      trainable=training)

    offset = tf.get_variable('offsets', initializer = tf.zeros([nPlanes], 
      tf.float32), trainable = False)

  return plane, offset

# Call the network created above on the query
query_features = create_network(q_img, size, First = True)

# Create the random vlalues
lsh_planes, lsh_offsets = generate_int_lsh_planes(query_features, nPlanes)

# Reshape to fit the limits for lsh application
query_features_shape = tf.reshape(query_features, [query_features.shape[0], 
  query_features.shape[1] * query_features.shape[2] * 
  query_features.shape[3]])

# Apply the lsh planes
query_lsh = tf.matmul(query_features_shape, lsh_planes)

support_list = []
query_list = []

# Go through each class and each support image in that class
for k in range(nClasses):
  slist=[]
  qlist=[]
  for i in range(nImgsSuppClass):
    support_result = create_network(s_imgs[:, k, i, :, :, :], size)
    # Fit the results to match the supports matrix multiplication
    support_shaped = tf.reshape(support_result, [support_result.shape[0],
      support_result.shape[1] * support_result.shape[2] * 
      support_result.shape[3]])

    # Apply the LSH Values
    support_lsh = tf.matmul(support_shaped, lsh_planes)
    support_lsh = tf.subtract(support_lsh, lsh_offsets)

    # This must be done so that we have a simple way to compare all supports
    # to one query
    slist.append(support_lsh)
    qlist.append(query_lsh)

  # Create tensorflow stack  
  slist = tf.stack(slist)
  qlist = tf.stack(qlist)
  support_list.append(slist)
  query_list.append(qlist)

# Make a stack to compare the query to every support
query_repeat = tf.stack(query_list)
supports = tf.stack(support_list)

# Loss
# LSH Calculation: multiplication of two vectors, use sigmoid to estimate
# 0 or 1 based on whether it is positive or negative
# Application of softmax  
# Minimize loss

# Logisitc k value
k = -1.0
with tf.name_scope("loss"):
  # Multiply the query by the supports
  #query_repeat = tf.Print(query_repeat, [query_repeat], summarize = 200, message = "query")
  #supports = tf.Print(supports, [supports], summarize = 200)
  signed = tf.multiply(query_repeat, supports)
  #signed = tf.Print(signed, [signed], summarize = 200)
  sigmoid = tf.sigmoid(signed)
  #sigmoid = tf.Print(sigmoid, [sigmoid], summarize = 200)

  # Sum the sigmoid values to ge the similarity
  similarity = tf.reduce_sum(sigmoid, [3])
  similarity = tf.transpose(similarity,[2,0,1])

  # Average the similarites among each class
  mean_similarity = tf.reduce_mean(similarity, 2)
  # Find the maximum similarty in each class
  max_similarity = tf.reduce_max(similarity, 2)
 
  #softmax = tf.nn.softmax(mean_similarity)
  #softmax = tf.Print(softmax, [softmax, tf.log(softmax + tf.constant(1e-10))], summarize = 300)
  #loss = -tf.reduce_sum(tf.cast(q_label, tf.float32) * tf.log(softmax + tf.constant(1e-10)))
  q_label = tf.stop_gradient(q_label)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=mean_similarity, labels=q_label)
  loss = tf.reduce_sum(cross_entropy)

# Optimizer
with tf.name_scope("optimizer"):
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Accuracy and Equality Distribution

with tf.name_scope("accuracy"):
  # Find the closest class
  max_class = tf.argmax(max_similarity, 1)
  # Find which class was supposed to be the closest
  max_label = tf.argmax(q_label, 1)  

  # Compare the values
  total = tf.equal(max_class, max_label)
  # Find, on average, how many were correct 
  accuracy = tf.reduce_mean(tf.cast(total, tf.float32))

def get_next_batch(test = False):
  suppImgs = []
  suppLabels = []
  # Get support values for each batch  
  for j in range(batchS):
    suppImgsOne = get_support(test)
    suppImgs.append(suppImgsOne)
  suppImgs = np.asarray(suppImgs)
  # Get query value for each batch
  queryImgBatch = []
  queryLabelBatch = []
  for i in range(batchS):
    qImg, qLabel = get_query(test)
    queryImgBatch.append(qImg)
    queryLabelBatch.append(qLabel)
  queryLabelBatch = np.asarray(queryLabelBatch)
  queryImgBatch = np.asarray(queryImgBatch)

  return suppImgs, suppLabels, queryImgBatch, queryLabelBatch

# Session

# Initialize the variables we start with
init = tf.global_variables_initializer()

with tf.Session() as session:
  session.run(init)
  
  # Create a save location
  Saver = tf.train.Saver()

  step = 1
  while step < nIt:
    step = step + 1

    suppImgs, suppLabels, queryImgBatch, queryLabelBatch = get_next_batch()
    
    # Run the session with the optimizer
    ACC, LOSS, OPT = session.run([accuracy, loss, optimizer], feed_dict
      ={s_imgs: suppImgs, 
        q_img: queryImgBatch,
        q_label: queryLabelBatch,
       })
    
    # Observe Values
    if (step%100) == 0:
      print("ITER: "+str(step))
      print("ACC: "+str(ACC))
      print("LOSS: "+str(LOSS))
      print("------------------------")
 
    # Run an additional test set 
    if (step%check) == 0:
      TotalAcc=0.0
      #run ten batches to test accuracy
      BatchToTest=10
      for repeat in range(BatchToTest):

        suppImgs, suppLabels, queryImgBatch, queryLabelBatch = get_next_batch(True)
          
        # Run session for test values
        ACC, LOSS = session.run([accuracy, loss], feed_dict
        ={s_imgs: suppImgs, 
          q_img: queryImgBatch,
          q_label: queryLabelBatch,
        })
        TotalAcc += ACC
      print("Accuracy on the independent test set is: "+str(TotalAcc/float(BatchToTest)) )

  # Save out the model once complete
  save_path = Saver.save(session, SAVE_PATH, step)
  print("Model saved in path: %s" % SAVE_PATH)
