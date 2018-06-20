# Using Cosine Distance to train a matching network

import tensorflow as tf
import numpy as np
import pickle
import random
import math
import sys
import os
import matplotlib.pyplot as plt

train_file_path = "../../../testing-data/cifar/data_batch_"
train_images_raw = np.empty((0, 3072))
train_labels_raw = np.empty((0))
for i in range(1,6):
  train_file_name = train_file_path + str(i)
  print(train_file_name)
  with open(train_file_name, 'rb') as cifar_file:
    data = pickle.load(cifar_file, encoding = 'bytes')
    train_images_raw = np.concatenate((train_images_raw, data[b"data"]), axis = 0)
    train_labels_raw = np.concatenate((train_labels_raw, data[b"labels"]), axis = 0)

test_file_name = "../../../testing-data/cifar/test_batch"
with open(test_file_name, 'rb') as cifar_file:
  test = pickle.load(cifar_file, encoding = 'bytes')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Graph Constants
size = [32, 32, 3]
nKernels = [8, 16, 32]
fully_connected_nodes = 128
poolS = 2

# Training information
nIt = 3000
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

if len(sys.argv) > 1 and sys.argv[1] != "-":
    base = sys.argv[1] + "/cosine-"
else:
    base = "/tmp/cifar-cosine-"

SAVE_PATH = base + str(nClasses)


train_images = []
train_labels = []
list_range = np.arange(len(train_images_raw))
np.random.shuffle(list_range)
for index, i in enumerate(list_range):
  train_images.append(train_images_raw[i])
  train_labels.append(train_labels_raw[i])

test_images = test[b"data"]
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
        imgReshape = np.reshape(test_images[imageNum,:], [3,32,32])
        imgReshape = np.transpose(imgReshape, [1,2,0])
        pickedLabels.append(test_labels[imageNum])
      pickedImages.append(imgReshape)
      samples += 1
    imageNum += 1
  return pickedImages, pickedLabels

# Get several images
def get_support(test=False):
  supportImgs = []
  
  if test:
    choices = numbersTest
  else:
    choices = numbers
  
  for support in choices:
    newSupportImgs, newSupportLabels = get_samples(support, nImgsSuppClass,
      test)
    supportImgs.append(newSupportImgs)
  
  return supportImgs

# Get a single query value
def get_query(test=False):
  if test:
    choices = numbersTest
  else:
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
    W = tf.get_variable('W',[FeatureLength,128])
    FC = tf.matmul(FC, W)
    Bias = tf.get_variable('Bias',[128])
    FC = tf.add(FC, Bias)
    FC = tf.reshape(FC, [batchS,128,1,1])
  
  return FC

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
  
    suppImgs = []

    # Get support values for each batch  
    for j in range(batchS):
      suppImgsOne = get_support(False)
      suppImgs.append(suppImgsOne)
    suppImgs = np.asarray(suppImgs)

    # Get query value for each batch
    queryImgBatch = []
    queryLabelBatch = []
    for i in range(batchS):
      qImg, qLabel = get_query(False)
      queryImgBatch.append(qImg)
      queryLabelBatch.append(qLabel)
    queryLabelBatch = np.asarray(queryLabelBatch)
    queryImgBatch = np.asarray(queryImgBatch)

    # Run the session with the optimizer
    ACC, LOSS, OPT = session.run([accuracy, loss, optimizer], feed_dict =
      {s_imgs: suppImgs, 
       q_img: queryImgBatch,
       q_label: queryLabelBatch
      })

    # Observe Values
    if (step%100) == 0:
      print("ITER: "+str(step))
      print("ACC: "+str(ACC))
      print("LOSS: "+str(LOSS))
      print("------------------------")
 
    # Run an additional test set
    if (step%1000) == 0:
      TotalAcc=0.0
      #run ten batches to test accuracy
      BatchToTest=10
      for repeat in range(BatchToTest):
	      suppImgs = []

        # Get supports	  
	      for j in range(batchS):
	      	suppImgsOne = get_support(False)
	      	suppImgs.append(suppImgsOne)
	      suppImgs = np.asarray(suppImgs)

        # Get queries
	      queryImgBatch = []
	      queryLabelBatch = []
	      for i in range(batchS):
	      	qImg, qLabel = get_query(False)
	      	queryImgBatch.append(qImg)
	      	queryLabelBatch.append(qLabel)
	      queryLabelBatch = np.asarray(queryLabelBatch)
	      queryImgBatch = np.asarray(queryImgBatch)

        # Run session for test values
	      ACC, LOSS = session.run([accuracy, loss], feed_dict = 
          {s_imgs: suppImgs, 
		       q_img: queryImgBatch,
		       q_label: queryLabelBatch
		      })

	      TotalAcc+=ACC
        
      print("Accuracy on the independent test set is: "+
        str(TotalAcc/float(BatchToTest)))

  # Save out the model once complete
  save_path = Saver.save(session, SAVE_PATH, step)
  print("Model saved in path: %s" % SAVE_PATH)
  
  # Use the test set
  sumAcc = 0.0
  for k in range(0,100):
    suppImgs = []
  
    # Get test support values 
    for j in range(batchS):
      suppImgsOne = get_support(True)
      suppImgs.append(suppImgsOne)
    suppImgs = np.asarray(suppImgs)
  
    # Get test queries
    queryImgBatch = []
    queryLabelBatch = []
    for i in range(batchS):
      qImg, qLabel = get_query(True)
      queryImgBatch.append(qImg)
      queryLabelBatch.append(qLabel)
      
    queryLabelBatch = np.asarray(queryLabelBatch)
    queryImgBatch = np.asarray(queryImgBatch)
    a = session.run(accuracy, feed_dict = {s_imgs: suppImgs, 
                                           q_img: queryImgBatch,
                                           q_label: queryLabelBatch
                                           })
    sumAcc += a
    
  print("Independent Test Set: "+str(float(sumAcc)/100))
