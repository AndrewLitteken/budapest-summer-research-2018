# Using One versus All LSH training in a matching network OMNIGLOT

import tensorflow as tf
import numpy as np
from scipy import misc
from skimage import transform, io
import random
import math
import sys 
import os
import scipy.misc
from sklearn import svm

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

train_images, test_images = make_dir_list(train_file_path)

# Graph Constants
size = [28, 28, 1]
nKernels = [8, 16, 32]
fully_connected_nodes = 128
poolS = 2

#Training information
nIt = 5000
if len(sys.argv) > 5 and sys.argv[5] != "-":
  nIt = int(sys.argv[5])
check = 1000
batchS = 32
nRandPlanes = 100
learning_rate = 1e-5

period = 1000
if len(sys.argv) > 4 and sys.argv[4] != "-":
  period = int(sys.argv[4])

# Support and testing infromation
nClasses = 5 
if len(sys.argv) > 2 and sys.argv[2] != "-":
  nClasses = int(sys.argv[2])
nImgsSuppClass = 5 
if len(sys.argv) > 3 and sys.argv[3] != "-":
  nImgsSuppClass = int(sys.argv[3])

if len(sys.argv) > 1 and sys.argv[1] != "-":
    base = sys.argv[1] + "/omniglot-lsh-one-all-"
else:
    base = "/tmp/omniglot-lsh-one-all-"

SAVE_PATH = base + str(period) + "-" + str(nClasses) + "-" + str(nImgsSuppClass)

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
  while len(selected_samples) < nImgsSuppClass:
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
    newSupportImgs = get_samples(choice, 1)
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

# Plane information: size of final layer, number of planes
lsh_planes = tf.placeholder(tf.float32, [None, None])
lsh_offsets = tf.placeholder(tf.float32, [None])

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

  return currInp

# Call the network created above on the query
query_features = create_network(q_img, size, First = True)

# Reshape to fit the limits for lsh application
query_features_shape = tf.reshape(query_features, [query_features.shape[0],
  query_features.shape[1] * query_features.shape[2] * 
  query_features.shape[3]])

# Apply LSH Matrix
query_lsh = tf.matmul(query_features_shape, lsh_planes)
query_lsh = tf.subtract(query_lsh, lsh_offsets)

# Empty Lists
support_list = []
query_list = []

support_features = []

# Iterate through each class and each support image in that class
for k in range(nClasses):
  slist=[]
  qlist=[]
  for i in range(nImgsSuppClass):
    support_result = create_network(s_imgs[:, k, i, :, :, :], size)
    # Fit the results to match the supports matrix multiplication
    support_shaped = tf.reshape(support_result, [support_result.shape[0], 
      support_result.shape[1] * support_result.shape[2] *
      support_result.shape[3]])
    # For access in the session
    support_features.append(support_shaped)

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
support_feature_vectors = tf.stack(support_features)
query_repeat = tf.stack(query_list)
supports = tf.stack(support_list)

# Loss
# LSH Calculation: multiplication of two vectors, use sigmoid to estimate
# 0 or 1 based on whether it is positive or negative
# Application of softmax  i
# Minimize loss

# Logisitc k value
k = -1.0
with tf.name_scope("loss"):
  # Multiply the query by the supports
  signed = tf.multiply(query_repeat, supports)

  # Apply a sigmoid function
  sigmoid = tf.divide(tf.constant(1.0),
    tf.clip_by_value(tf.add(tf.constant(1.0),
    tf.exp(tf.multiply(tf.constant(k), signed))),
    1e-10, float("inf")))
  
  # Sum the sigmoid values to ge the similarity
  similarity = tf.reduce_sum(sigmoid, [3])
  similarity = tf.transpose(similarity,[2,0,1])
  
  # Average the similarities among each class
  mean_similarity = tf.reduce_mean(similarity, 2)
  # Find the maximum similarity in each class
  max_similarity = tf.reduce_max(similarity, 2)
  
  # Use softmax against the query label to compare for an expected
  # distribution 
  loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=mean_similarity, labels=q_label))

# Optimizer, Adam Optimizer as it seems to work the best
with tf.name_scope("optimizer"):
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Accuracy and Equality Distribution

with tf.name_scope("accuracy"):
  # Find the closest class
  max_class = tf.argmax(max_similarity, 1)
  # Find which class was supposed to be the closes
  max_label = tf.argmax(q_label, 1)  
  
  # Compare the values
  total = tf.equal(max_class, max_label) 
  # Find, one average, how many were correct
  accuracy = tf.reduce_mean(tf.cast(total, tf.float32))

# Pick the planes using an Support Vector Matrix method
def gen_lsh_pick_planes(nPlanes, feature_vectors, labels):
  lsh_matrix = []
  lsh_offset_vals = []
  
  feature_vectors = np.reshape(np.asarray(feature_vectors),
    (len(feature_vectors), -1))

  label_dirs = set(np.reshape(labels, -1))
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
    lsh_offset_vals.append(temp_mul)

  lsh_matrix = np.transpose(lsh_matrix)
  return lsh_matrix, lsh_offset_vals

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

def remake_planes(suppLabels, SFV):
  new_labels = []
  for i in range(len(suppLabels)):
    new_labels.append([])
    for j in range(len(suppLabels[0])):
      for k in range(nImgsSuppClass):
        new_labels[i].append(suppLabels[i][j])  
     
  new_labels = np.asarray(new_labels)

  # Reshape the support vectos for our use
  SFV = np.transpose(SFV, (1, 0, 2))
  new_feature_vectors = np.reshape(SFV, (SFV.shape[0]*SFV.shape[1], -1))
  labels = np.reshape(new_labels, (new_labels.shape[0] * new_labels.shape[1], -1))
  # Pick the planes based off of the sprad created by the initial network
  planes, offsets = gen_lsh_pick_planes(10, new_feature_vectors, labels)

  return planes, offsets
# Session

# Initialize the vairables we start with
init = tf.global_variables_initializer()

with tf.Session() as session:
  session.run(init)
  
  # Create a save location
  Saver = tf.train.Saver()  

  suppImgs, suppLabels, queryImgBatch, queryLabelBatch = get_next_batch()

  # Generate some random planes with slope values between 1 and -1
  random_planes = (np.matlib.rand(fully_connected_nodes, nRandPlanes) 
    - 0.5) * 2
  # Generate blank offsets
  blank_offsets = np.zeros(nRandPlanes)

  # Run the session with these planes, the planes do not truly affect the
  # result it is just to keep everything simpler
  SFV, QF = session.run([support_feature_vectors, query_features], feed_dict
    ={s_imgs: suppImgs, 
      q_img: queryImgBatch,
      q_label: queryLabelBatch,
      lsh_planes: random_planes,
      lsh_offsets: blank_offsets
     })

  planes, offsets = remake_planes(suppLabels, SFV)

  step = 1
  while step < nIt:
    step = step + 1

    suppImgs, suppLabels, queryImgBatch, queryLabelBatch = get_next_batch()
    
    # Run the session with the optimizer
    SFV, ACC, LOSS, OPT = session.run([support_feature_vectors, accuracy, 
      loss, optimizer], feed_dict
      ={s_imgs: suppImgs, 
        q_img: queryImgBatch,
        q_label: queryLabelBatch,
        lsh_planes: planes,
        lsh_offsets: offsets
       })

    # Rework the planes ever period iterations
    if (step % period) == 0:
      planes, offsets = remake_planes(suppLabels, SFV)
    
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

        suppImgs, suppLabels, queryImgBatch, queryLabelBatch = get_next_batch(True)
          
        # Run session for test values
        ACC, LOSS = session.run([accuracy, loss], feed_dict
        ={s_imgs: suppImgs, 
          q_img: queryImgBatch,
          q_label: queryLabelBatch,
          lsh_planes: planes,
          lsh_offsets: offsets
        })
        TotalAcc += ACC
      print("Accuracy on the independent test set is: "+str(TotalAcc/float(BatchToTest)) )
 
  # Save out the model once complete
  save_path = Saver.save(session, SAVE_PATH, step)
  print("Model saved in path: %s" % SAVE_PATH)