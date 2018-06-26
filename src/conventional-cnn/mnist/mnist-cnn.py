# Using Cosine Distance to train a matching network

import tensorflow as tf
import numpy as np
import random
import math
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../testing-data/MNIST_data/",
  one_hot=True)


train_images = mnist.train.images
train_labels = mnist.train.labels

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Graph Constants
size = [28, 28, 1]
nKernels = [8, 16, 32]
fully_connected_nodes = 128
poolS = 2

# Training information
nIt = 10000
batchS = 32
learning_rate = 1e-4

# Support and testing infromation
classList = [1,2,3,4,5,6,7,8,9,0]
numbers = [0,1,2]
numbersTest = [7,8,9]
nClasses = len(numbers)
if len(sys.argv) > 2:
  nClasses = int(sys.argv[2])
  numbers = classList[:nClasses]
  numbersTest = classList[10-nClasses:]
nClasses = len(numbers)
nImgsSuppClass = 5

if len(sys.argv) > 1:
    base = sys.argv[1] + "/mnist-cnn-"
else:
    base = "/tmp/mnist-cnn-"

SAVE_PATH = base + str(nClasses)

# Query Information - vector
q_img = tf.placeholder(tf.float32, [batchS]+size) # batch size, size
# batch size, number of categories
q_label = tf.placeholder(tf.int32, [batchS, len(numbers)])

# Network Function
# Call for each support image (row of the support matrix) and for the  query 
# image.

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
      convR = tf.nn.conv2d(currInp, weight, strides=[1,1,1,1], padding="SAME")
      convR = tf.add(convR, bias)
      reluR = tf.nn.relu(convR)
      poolR = tf.nn.max_pool(reluR, ksize=[1,poolS,poolS,1], 
        strides=[1,poolS,poolS,1], padding="SAME")
      currInp = poolR
  
  with tf.variable_scope('FC', reuse = tf.AUTO_REUSE) as varscope:
    CurrentShape=currInp.get_shape()
    FeatureLength = int(CurrentShape[1]*CurrentShape[2]*CurrentShape[3])
    FC = tf.reshape(currInp, [-1,FeatureLength])
    W = tf.get_variable('W',[FeatureLength,len(numbers)])
    FC = tf.matmul(FC, W)
    Bias = tf.get_variable('Bias',[len(numbers)])
    FC = tf.add(FC, Bias)
    FC = tf.reshape(FC, [batchS,len(numbers)])
  
  return FC

# Call the network created above on the qury
network_result = create_network(q_img, size)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = q_label, logits = network_result))
# Optimizer
with tf.name_scope("optimizer"):
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Accuracy and Equality Distribution

with tf.name_scope("accuracy"):
  correct_predictions = tf.equal(tf.argmax(network_result, 1), tf.argmax(q_label, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

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
    queryImgBatch = []
    queryLabelBatch = []
    occurrences = np.zeros(len(numbers))
    for i in range(batchS):
      selected_class = random.randint(0, (len(numbers) - 1))
      while occurrences[selected_class] > 0 and 0 in occurrences:
        selected_class = random.randint(0, (len(numbers) - 1))

      occurrences[selected_class] += 1

      selected_sample = random.randint(0, len(train_images) - 1)
      while np.argmax(train_labels[selected_sample]) != selected_class:
        selected_sample += 1
        if selected_sample == len(train_images):
          selected_sample = 0
      qImg = train_images[selected_sample]
      qLabel_int = int(np.argmax(train_labels[selected_sample]))
      qLabel = np.zeros(len(numbers))
      qLabel[qLabel_int] = 1

      queryImgBatch.append(qImg)
      queryLabelBatch.append(qLabel)
    queryLabelBatch = np.asarray(queryLabelBatch)
    queryImgBatch = np.asarray(queryImgBatch)
    queryImgBatch = np.reshape(queryImgBatch, [batchS] + size)

    # Run the session with the optimizer
    ACC, LOSS, OPT = session.run([accuracy, loss, optimizer], feed_dict =
      {q_img: queryImgBatch,
       q_label: queryLabelBatch
      })

    # Observe Values
    if (step%100) == 0:
      print("ITER: "+str(step))
      print("ACC: "+str(ACC))
      print("LOSS: "+str(LOSS))
      print("--------------------")

  save_path = Saver.save(session, SAVE_PATH, step)
  print("Model saved in path: %s" % SAVE_PATH)
