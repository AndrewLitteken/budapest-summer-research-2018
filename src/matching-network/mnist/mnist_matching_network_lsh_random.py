# Using Random LSH Planes training in a matching network MNIST

import tensorflow as tf
import numpy as np
import getopt
import random
import math
import sys
import os

# Import data, to be replaced with more flexible importing
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../../testing-data/MNIST_data/",
  one_hot=True)

# Hardware Specifications
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def help_message():
  exit(1)

# Graph Constants
size = [28, 28, 1]
nKernels = [64, 64,64]
fully_connected_nodes = 128
poolS = 2

#Training information
nIt = 5000

check = 1000
batchS = 32
nPlanes = 100
learning_rate = 1e-5

# Support and testing information
classList = [1,2,3,4,5,6,7,8,9,0]
numbers = []
numbersTest = []

nClasses = 3
nImgsSuppClass = 5
nPlanes = 100
tensorboard = False
batch_norm = False
dropout = False
integer_range = 100

training = False

base = "/tmp/minst-lsh-random-"

opts, args = getopt.getopt(sys.argv[1:], "hmnodtr:L:c:p:i:b:s:", ["help", 
  "num_classes=", "num_supports=", "num_planes=", "base_path=", 
  "num_iterations=", "integer_range=", "training","meta_tensorboard",
  "dropout", "batch_norm", "num_layers="])

for o, a in opts:
  if o in ("-t", "--training"):
    training = True
  elif o in ("-c", "--num_classes"):
    nClasses = int(a)
  elif o in ("-s", "--num_supports"):
    nImgsSuppClass = int(a)
  elif o in ("-b", "--base_path"):
    base = a + "mnist-lsh-random-"
  elif o in ("-p", "--num_planes"):
    nPlanes = int(a)
  elif o in ("-i", "--num_iterations"):
    nIt = int(a)
  elif o in ("-r", "--integer_range"):
    integer_range = int(a)
  elif o in ("-m", "--meta_tensorboard"):
    tensorboard = True
  elif o in ("-d", "--data_dir"):
    train_file_path = "../../../testing-data/omniglot-rotate/"
  elif o in ("-o", "--dropout"):
    dropout = True
  elif o in ("-n", "--batch_norm"):
    batch_norm = True
  elif o in ("-L", "--num_layers"):
    nKernels = [64 for x in range(int(a))]
  elif o in ("-h", "--help"):
    help_message()
  else:
    print("unhandled option: "+o)
    help_message()

numbers = classList[:nClasses]
numbersTest = classList[10-nClasses:]

SAVE_PATH = base
if batch_norm:
  SAVE_PATH += "norm-"
if dropout:
  SAVE_PATH += "dropout-"

SAVE_PATH += str(len(nKernels)) + "-" + str(training) + "-" + str(nPlanes) + "-" + str(nClasses) + "-" + str(nImgsSuppClass)

LOG_DIR = "./mnist/lsh_random/"
if batch_norm:
  LOG_DIR += "norm/"
if dropout:
  LOG_DIR += "dropout/"
LOG_DIR += str(len(nKernels)) + "/" + str(training) + "/" + str(nPlanes) + "/" + str(nClasses) + "/" + str(nImgsSuppClass)

# Collecting sample both for query and for testing
def get_samples(mnistNum, nSupportImgs, testing = False):
  one_hot_list = [0.] * 10
  one_hot_list[mnistNum] = 1.
  samples = 0
  if not testing:
    imageNum = random.randint(0, mnist.train.images.shape[0] - 1)
  else:
    imageNum = random.randint(0, mnist.test.images.shape[0] - 1)
  pickedImages = []
  pickedLabels = []
  while samples < nSupportImgs:
    if (imageNum == len(mnist.train.images) and not testing):
      imageNum = 0
    elif (imageNum == len(mnist.test.images) and testing):
      imageNum = 0
    if not testing:
      labelThis = mnist.train.labels[imageNum, :]
    else:
      labelThis = mnist.test.labels[imageNum, :]
    if np.all(labelThis == one_hot_list):
      if not testing:
        imgReshape = np.reshape(mnist.train.images[imageNum,:], size)
        pickedLabels.append(mnist.train.labels[imageNum, :])
      else:
        imgReshape = np.reshape(mnist.test.images[imageNum,:], size)
        pickedLabels.append(mnist.test.labels[imageNum, :])
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
  img, label = get_samples(imageNum, 1, test)
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
        reuse=tf.AUTO_REUSE) as varscope, tf.name_scope('conv'+str(layer)):
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
      tf.float32))

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
  signed = tf.multiply(query_repeat, supports)
  sigmoid = tf.sigmoid(signed)

  # Sum the sigmoid values to ge the similarity
  similarity = tf.reduce_sum(sigmoid, [3])
  similarity = tf.transpose(similarity,[2,0,1])

  # Average the similarites among each class
  mean_similarity = tf.reduce_mean(similarity, 2)
  # Find the maximum similarty in each class
  max_similarity = tf.reduce_max(similarity, 2)
 
  loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=mean_similarity, labels=q_label))

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
    # Run the session with the optimizer
    if tensorboard and step == 2:
      writer = tf.summary.FileWriter(LOG_DIR + "/" + str(step), session.graph)
      runOptions = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      ACC, LOSS, OPT = session.run([accuracy, loss, optimizer], feed_dict
        ={s_imgs: suppImgs, 
          q_img: queryImgBatch,
          q_label: queryLabelBatch,
         }, options = runOptions, run_metadata=run_metadata)
      writer.add_run_metadata(run_metadata, 'step%d' % i)
    else:
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
  
  # Use the test set
  '''sumAcc = 0.0
  for k in range(0,100):
  
    # Get test support values 
    suppImgs, suppLabels, queryImgBatch, queryLabelBatch = get_next_batch(True)

    a = session.run(accuracy, feed_dict = {s_imgs: suppImgs, 
                                           q_img: queryImgBatch,
                                           q_label: queryLabelBatch
                                           })
    sumAcc += a
    
  print("Independent Test Set: "+str(float(sumAcc)/100))'''
