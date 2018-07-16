# Using Cosine Distance to train a matching network omniglot

from skimage import transform, io
import tensorflow as tf
import numpy as np
import scipy.misc
import getopt
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
nKernels = [64, 64, 64]
fully_connected_nodes = 128
poolS = 2

# Training information
nIt = 5000
check = 1000
batchS = 32
learning_rate = 1e-4

# Support and testing infromation
nClasses = 3
nImgsSuppClass = 5

base = "/tmp/omniglot-cosine-"

opts, args = getopt.getopt(sys.argv[1:], "hdc:i:b:s:", ["help", 
  "num_classes=", "num_supports=", "base_path=", "num_iterations="])

for o, a in opts:
  if o in ("-c", "--num_classes"):
    nClasses = int(a)
  elif o in ("-s", "--num_supports"):
    nImgsSuppClass = int(a)
  elif o in ("-b", "--base_path"):
    base = a + "omniglot-cosine-"
  elif o in ("-i", "--num_iterations"):
    nIt = int(a)
  elif o in ("-d", "--data"):
    train_file_path = "../../../testing-data/omniglot-rotate/"
  elif o in ("-h", "--help"):
    help_message()
  else:
    print("unhandled option: "+o)
    help_message()

train_images, test_images = make_dir_list(train_file_path)

SAVE_PATH = base + str(nClasses) + "-" + str(nImgsSuppClass)

train_dirs, test_dirs = make_dir_list(train_file_path) 

# Collecting sample both for query and for testing
def get_samples(data_dir, nSupportImgs):
  
  img_names = []
  for file_name in os.listdir(data_dir):
    img_names.append("{}/{}".format(data_dir, file_name))

  img_names = np.asarray(img_names)
  np.random.shuffle(img_names)

  selected_samples = []
  picked_images = []
  while len(selected_samples) < nSupportImgs:
    picked_image_name = random.choice(img_names)
    while picked_image_name in selected_samples:
      picked_image_name = random.choice(img_names)
    picked_image = io.imread(picked_image_name)
    image_resize = transform.resize(picked_image, size)
    picked_images.append(image_resize)
    selected_samples.append(picked_image_name)

  return picked_images

# Get several images
def get_support(test=False):
  supportImgs = []
  
  choices = train_images
  
  characters = []
  while len(characters) < nClasses:
    choice = random.choice(choices)
    while choice in characters:
      choice = random.choice(choices)
    characters.append(choice)
    newSupportImgs = get_samples(choice, nImgsSuppClass)
    supportImgs.append(newSupportImgs)
  
  return supportImgs, characters

# Get a single query value
def get_query(available_chars, test=False):
  imageInd = random.randint(0, len(available_chars) - 1) 
  image_name = available_chars[imageInd]
  img = get_samples(image_name, 1)
  l=np.zeros(len(available_chars))
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
