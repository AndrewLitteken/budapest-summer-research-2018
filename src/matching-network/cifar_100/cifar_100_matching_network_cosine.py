# Using Cosine Distance to train a matching network CIFAR-100

import tensorflow as tf
import numpy as np
import getopt
import pickle
import random
import math
import sys
import os

import cifar_100

cifar_100.get_data()

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

# Training Infomration
nIt = 5000
check = 1000
batchS = 32
nRandPlanes = 100
learning_rate = 1e-6
tensorboard = False
# Support and testing information
nClasses = 3
batch_norm = False
batch_norm = True
nImgsSuppClass = 5

base = "/tmp/cifar-100-cosine-"

opts, args = getopt.getopt(sys.argv[1:], "hmnodL:c:i:b:s:", ["help", 
  "num_classes=", "num_supports=", "base_path=", "num_iterations=",
  "dropout", "batch_norm", "num_layers="])

for o, a in opts:
  if o in ("-c", "--num_classes"):
    nClasses = int(a)
  elif o in ("-s", "--num_supports"):
    nImgsSuppClass = int(a)
  elif o in ("-b", "--base_path"):
    base = a
    if a[-1] != "/":
      base += "/"
    base += "cifar-100-cosine-"
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
  elif o in ("-L", "--num_layers"):
    nKernels = [64 for x in range(int(a))]
  elif o in ("-h", "--help"):
    help_message()
  else:
    print("unhandled option: "+o)
    help_message()

SAVE_PATH = base 
if batch_norm:
  SAVE_PATH += "norm-"
if dropout:
  SAVE_PATH += "dropout-"
SAVE_PATH += str(len(nKernels)) + "-" + str(nClasses) + "-" + str(nImgsSuppClass)

LOG_DIR = "./cifar_100_network_training/cosine/"
if batch_norm:
  LOG_DIR += "norm/"
if dropout:
  LOG_DIR += "dropout/"
LOG_DIR += str(len(nKernels)) + "/" + str(nClasses) + "/" + str(nImgsSuppClass) 

train_images = []
train_labels = []
list_range = np.arange(len(train_images_raw))
np.random.shuffle(list_range)
for index, i in enumerate(list_range):
  if train_labels_raw[i] < 80:
    train_images.append(train_images_raw[i])
    train_labels.append(train_labels_raw[i])

test_images = test[b"data"]
test_labels = test[b"fine_labels"]

# Collecting sample both for query and for testing
def get_samples(class_num, nSupportImgs):
  selected_samples = []
  
  picked_images = []
  while len(selected_samples) < nSupportImgs:
    picked_index = random.randint(0, len(train_labels) - 1)
    while (class_num != train_labels[picked_index] or 
      picked_index in selected_samples):
      picked_index = random.randint(0, len(train_labels) - 1)
    imgReshape = np.reshape(train_images[picked_index], [3,32,32])
    imgReshape = np.transpose(imgReshape, [1,2,0])
    picked_images.append(imgReshape)
    selected_samples.append(picked_index)

  return picked_images

# Get several images
def get_support(test=False):
  supportImgs = []
  
  choices = train_labels
  
  images = []
  while len(images) < nClasses:
    choice = random.choice(choices)
    while choice in images:
      choice = random.choice(choices)
    images.append(choice)
    newSupportImgs = get_samples(choice, nImgsSuppClass)
    supportImgs.append(newSupportImgs)
  
  return supportImgs, images

# Get a single query value
def get_query(available_images, test=False):
  choices = available_images
  imageInd = random.randint(0, len(choices) - 1)
  imageNum = choices[imageInd]
  img = get_samples(imageNum, 1)
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
q_label = tf.placeholder(tf.int32, [batchS, None])

# Network Function
# Call for each support image (row of the support matrix) and for the  query 
# image.

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

# Call the network created above on the qury
query_features = create_network(q_img, size, First = True)

support_list = []
query_list = []

# Go through each class and each support image in that class
for k in range(nClasses):
  slist=[]
  qlist=[]
  for i in range(nImgsSuppClass):
    slist.append(create_network(s_imgs[:, k, i, :, :, :], size))
    qlist.append(query_features)
  slist = tf.stack(slist)
  qlist = tf.stack(qlist) 
  support_list.append(slist)
  query_list.append(qlist)

# Make a stack to compare the query to every support
query_repeat = tf.stack(query_list)
supports = tf.stack(support_list)

# Loss
# Cosine distance calculation  
# Application of softmax  
# Minimize loss

with tf.name_scope("loss"):
  dotProduct = tf.reduce_sum(tf.multiply(query_repeat, supports), [3,4,5])
  supportsMagn = tf.sqrt(tf.reduce_sum(tf.square(supports), [3,4,5]))
  cosDist = dotProduct / tf.clip_by_value(supportsMagn, 1e-10, float("inf"))
  
  cosDist = tf.transpose(cosDist,[2,0,1])

  # Find the average cosine distance per class
  MeanCosDist= tf.reduce_mean(cosDist,2)
  # fnd the maximum cosine distance per class
  MaxCostDist = tf.reduce_max(cosDist,2)

  loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits = MeanCosDist, labels = q_label))

# Optimizer
with tf.name_scope("optimizer"):
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Accuracy and Equality Distribution

with tf.name_scope("accuracy"):
  # Find the closest class
  max_class = tf.argmax(MaxCostDist, 1)
  # Find whihc class was supposed to be the closest
  max_label = tf.argmax(q_label, 1)  
  
  # Compare the values
  total = tf.equal(max_class, max_label) 
  # Find on average, how many were correct
  accuracy = tf.reduce_mean(tf.cast(total, tf.float32))

def get_next_batch(test = False):
  suppImgs = []
  suppLabels = []
  # Get support values for each batch  
  for j in range(batchS):
    suppImgsOne, suppLabelsOne = get_support(test)
    suppImgs.append(suppImgsOne)
    suppLabels.append(suppLabelsOne)
  suppImgs = np.asarray(suppImgs)
  suppLabels = np.asarray(suppLabels)
  # Get query value for each batch
  queryImgBatch = []
  queryLabelBatch = []
  for i in range(batchS):
    qImg, qLabel = get_query(suppLabels[i], test)
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
