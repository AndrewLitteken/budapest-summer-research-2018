# Using Cosine Distance to train a matching network

import tensorflow as tf
import numpy as np
from scipy import misc
import random
import math
import sys
import os

train_file_path = "../../../testing-data/omniglot/"

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
nKernels = [8, 16, 32]
fully_connected_nodes = 128
poolS = 2

# Training information
nIt = 3000
batchS = 32
learning_rate = 1e-4

# Support and testing infromation
nClasses = 5
if len(sys.argv) > 2 and sys.arv[2] != "-":
  nClasses = int(sys.argv[2])
nImgsSuppClass = 5

if len(sys.argv) > 1:
    base = sys.argv[1] + "/omniglot-cosine-"
else:
    base = "/tmp/omniglot-cosine-"

SAVE_PATH = base + str(nClasses)

train_dirs, test_dirs = make_dir_list(train_file_path) 

# Collecting sample both for query and for testing
def get_samples(data_dir, nSupportImgs):
  
  img_names = []
  for file_name in os.listdir(data_dir):
    img_names.append("{}{}".format(data_dir, file_name))

  img_names = np.asarray(img_names)
  selected_indices = np.arange(len(img_names))
  selected_indices = np.random.shuffle(selected_indices)[0:nSupportImgs]
  img_names = np.asarray(img_names)[selected_indices]

  picked_images = []

  return pickedImages

# Get several images
def get_support(test=False):
  supportImgs = []
  
  if test:
    choices = test_images
  else:
    choices = train_images
  
  for :
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
