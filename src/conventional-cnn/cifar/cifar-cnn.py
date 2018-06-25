# Using One versus All LSH training in a matching network

import tensorflow as tf
import numpy as np
import pickle
import random
import math
import sys
import os
from sklearn import svm

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

# Hardware specifications
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Graph Constants
size = [32, 32, 3]
nKernels = [8, 16, 32]
poolS = 2

# Training Infomration
nIt = 100000
batchS = 32
nRandPlanes = 100
learning_rate = 1e-4

# Support and testing information
classList = [1,2,3,4,5,6,7,8,9,0]
numbers = [1,2,3]
numbersTest = [8,9,0]
if len(sys.argv) > 2:
  nClasses = int(sys.argv[2])
  numbers = classList[:nClasses]
  numbersTest = classList[10-nClasses:]
nClasses = len(numbers)
nImgsSuppClass = 5

# Plane Training information
if len(sys.argv) > 1 and sys.argv[1] != "-":
    base = sys.argv[1] + "/cifar-cnn-"
else:
  base = "/tmp/cifar-cnn-"

SAVE_PATH = base + str(nClasses)

train_images = []
train_labels = []
list_range = np.arange(len(train_images_raw))
np.random.shuffle(list_range)
for index, i in enumerate(list_range):
  train_images.append(train_images_raw[i])
  train_labels.append(train_labels_raw[i])

train_images = np.reshape(train_images, [len(train_images)] + [3, 32, 32])
train_images = np.transpose(train_images, [0, 2, 3, 1]) 

test_images = test[b"data"]
test_images = np.reshape(test_images, [len(test_images)] + [3, 32, 32])
test_images = np.transpose(test_images, [0, 2, 3, 1]) 
test_labels = test[b"labels"]

tf.reset_default_graph()

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
  
  for k in nKernels:
    with tf.variable_scope('conv'+str(layer), 
      reuse=tf.AUTO_REUSE) as varscope:
      layer += 1
      weight = tf.get_variable('weight', [3,3,currFilt,k])
      currFilt = k
      bias = tf.get_variable('bias', [k], initializer = 
        tf.constant_initializer(0.0))
      convR = tf.nn.conv2d(currInp, weight, strides=[1,1,1,1],
        padding="SAME")
      convR = tf.add(convR, bias)
      reluR = tf.nn.relu(convR)
      poolR = tf.nn.max_pool(reluR, ksize=[1,poolS,poolS,1],
        strides=[1,poolS,poolS,1], padding="SAME")
      currInp = poolR
  
  with tf.variable_scope('FC', reuse = tf.AUTO_REUSE) as varscope:
    CurrentShape=currInp.get_shape()
    FeatureLength = int(CurrentShape[1]*CurrentShape[2]*CurrentShape[3])
    FC = tf.reshape(currInp, [-1,FeatureLength])
    W = tf.get_variable('W',[FeatureLength, len(numbers)])
    FC = tf.matmul(FC, W)
    Bias = tf.get_variable('Bias',[len(numbers)])
    FC = tf.add(FC, Bias)
  
  return FC

network_result = create_network(q_img, size)
# Loss
# LSH Calculation: multiplication of two vectors, use sigmoid to estimate
# 0 or 1 based on whether it is positive or negative
# Application of softmax  i
# Minimize loss

# Logisitc k value
k = -1.0
with tf.name_scope("loss"):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = q_label, logits = network_result))

# Optimizer, Adam Optimizer as it seems to work the best
with tf.name_scope("optimizer"):
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Accuracy and Equality Distribution

with tf.name_scope("accuracy"):
  correct_predictions = tf.equal(tf.argmax(network_result, 1), tf.argmax(q_label, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Session

# Initialize the vairables we start with
init = tf.global_variables_initializer()

with tf.Session() as session:
  session.run(init)
  
  # Create a save location
  Saver = tf.train.Saver()  
  
  step = 1
  while step < nIt:
    step = step + 1
    # Get query value for each batch
    queryImgBatch = []
    queryLabelBatch = []
    occurrences = np.zeros(len(numbers))
    for i in range(batchS):
      selected_class = random.randint(0, (len(numbers) - 1))
      while occurrences[selected_class] > 0 and 0 in occurrences:
        selected_class = random.randint(0, (len(numbers) - 1))
      
      occurrences[selected_class] += 1     
 
      selected_sample = random.randint(0, len(train_images) - 1) 
      while train_labels[selected_sample] != selected_class:
        selected_sample += 1
        if selected_sample == len(train_images):
          selected_sample = 0
      qImg = train_images[selected_sample]
      qLabel_int = int(train_labels[selected_sample])
      qLabel = np.zeros(len(numbers))
      qLabel[qLabel_int] = 1

      queryImgBatch.append(qImg)
      queryLabelBatch.append(qLabel)
    queryLabelBatch = np.asarray(queryLabelBatch)
    queryImgBatch = np.asarray(queryImgBatch)
    
    # Run the session with the optimizer
    ACC, LOSS, OPT = session.run([accuracy, loss, optimizer], feed_dict =
      {q_img: queryImgBatch,
        q_label: queryLabelBatch,
      })
    
    # Observe Values
    if (step%100) == 0:
      print("ITER: "+str(step))
      print("ACC: "+str(ACC))
      print("LOSS: "+str(LOSS))
      print("------------------------")

  save_path = Saver.save(session, SAVE_PATH, step)
  print("Model saved in path: %s" % SAVE_PATH)
 
  sumAcc = 0.0

  # Use the test set 
  for k in range(0,100):
    queryImgBatch = []
    queryLabelBatch = []
    occurrences = np.zeros(len(numbers))
    for i in range(batchS):
      selected_class = random.randint(0, (len(numbers) - 1))
      while occurrences[selected_class] > 0 and 0 in occurrences:
        selected_class = random.randint(0, (len(numbers) - 1))
      
      occurrences[selected_class] += 1     
 
      selected_sample = random.randint(0, len(test_images) - 1) 
      while test_labels[selected_sample] != selected_class:
        selected_sample += 1
        if selected_sample == len(test_images):
          selected_sample = 0
      qImg = test_images[selected_sample]
      qLabel_int = int(test_labels[selected_sample])
      qLabel = np.zeros(len(numbers))
      qLabel[qLabel_int] = 1

      queryImgBatch.append(qImg)
      queryLabelBatch.append(qLabel)
    queryLabelBatch = np.asarray(queryLabelBatch)
    queryImgBatch = np.asarray(queryImgBatch)
    
 
    
    a = session.run(accuracy, feed_dict = {q_img: queryImgBatch,
                                           q_label: queryLabelBatch,
                                          })
    sumAcc += a
    
  print("Independent Test Set: "+str(float(sumAcc)/100))
