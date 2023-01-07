#importing the needed library
from email.mime import image
from unittest import result
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# initialize the path and index of the train dataset
# Our dataset are used to train the program so that the program can distinguish body shapes which are divided into 3, namely: 
# 1. Over Weight, 
# 2. Normal, and 
# 3. Under Weight
# In this section, we want to initialize the path and the index of our dataset and add (append) it into the list for all of the picture in the
# dataset file
train_path = 'Dataset/Train'
train_path_list = os.listdir(train_path)
labels = train_path_list

image_list = []
image_classes_list = []

for index, class_path in enumerate(train_path_list):
    image_path_list = os.listdir(train_path + '/' + class_path)
    for image_path in image_path_list:
        image_list.append(train_path + '/' + class_path + '/' + image_path)
        image_classes_list.append(index)

# Extract Feature 
# We are using SIFT as our descriptor. As we know, SIFT is used to detect the interest points on an input image and 
# helps located the local features in an image that commonly known as the ‘keypoints‘ of the image. Therefore, we can identify of localized 
# features in images which is essential using SIFT.
sift = cv2.SIFT_create()
des_list = []

for image in image_list:
    # in the SIFT method, actually we will need to do the selection of key point(kp) and then make the key point descriptor(des), 
    # but because we don't want eliminate any key point of the picture, then we just doing the descriptor for all of the key point.
    # in this section, we try to compute the descriptor and append it into the list. because we don't want to detect the key point(kp), then 
    # it mark as "_" 
    _, des = sift.detectAndCompute(cv2.imread(image), None)
    des_list.append(des)

# Preprocessing
# on this section, we want to stacked the sequence of input arrays, so we just initiate the stacked_des variable with the first index (0) of des_list
stacked_des = des_list[0]

# then, we do a looping way to stacked the sequence of des_list arrays vertically to make a single array.
for des in des_list[1:]:
    stacked_des = np.vstack((stacked_des, des))

# in this section, we convert each value in the stacked_des array to be a float of size 32 bits
stacked_des = np.float32(stacked_des)

# Clustering
# in this section, we want to do some clustering for the stacked_des array. 
# we used "kmeans" method to doing the clustering for the dataset. As we know, kmeans is used to identify clusters of data objects in a dataset with 
# iteratively divides data points into K clusters by minimizing the variance in each cluster. 
# The clustering in kmeans will produce 2 value, there are centroid and distortion. Here, we just want to use the centroid part.
centroid, _ = kmeans(stacked_des, 3, 20)

# then, we want to created a new array of given shapes and types filled with zero values,
# therefore, we used zeros in numpy with parameters the length of array image_list (total of dataset) and length of centroid.
image_feature = np.zeros((len(image_list), len(centroid)), dtype= "float32")

# In this section, we will do some Vector Quantization for every dataset which we have. 
# We doing the Vector Quantization here to recognited the pattern and density estimation and clustering.
for i in range(len(image_list)):
    words, _ = vq(des_list[i], centroid)

    for w in words:
        image_feature[i][w] += 1

# in this section, we try to standardize and transform the image_feature values into a standard format, so we used StandardScaler.
std_scaler = StandardScaler().fit(image_feature)

# Then, we put the data which already transform using StandardScaler (std_scaler) into the image_feature variable.
image_feature = std_scaler.transform(image_feature)

# in this section, we will do some classifiying with separate the data linearly. Thefore, we used the LinearSVC function
svc = LinearSVC()
svc.fit(image_feature, np.array(image_classes_list))

# same as the training dataset, this process is intended to initialize the path of the test's dataset
# Test
test_path = 'Dataset/Test'
test_path_list = os.listdir(test_path)

image_list = []

for image in test_path_list:
    # Dataset/Test/1.jpeg
    image_list.append(test_path + '/' + image)

# in this process, the descriptor will be initialized by sift algorithm. Same as the previous process (training)
# basically, in this process there is only prepare the descriptor for preparing detect the test image
sift = cv2.SIFT_create()
des_list = []

for image in image_list:
    _, des = sift.detectAndCompute(cv2.imread(image), None)
    des_list.append(des)


# Testing
# Basically, in this process is intended as predicting the test data. and the process has a same vision with the previous
# In this section, we will do some Vector Quantization for every dataset which we have. 
# We doing the Vector Quantization here to recognited the pattern and density estimation and clustering.
test_feature = np.zeros((len(image_list), len(centroid)), "float32")

for i in range(len(image_list)):
    words, _ = vq(des_list[i], centroid)

    for w in words:
        test_feature[i][w] += 1

test_feature = std_scaler.transform(test_feature)
result = svc.predict(test_feature)

# Show Result
# And then in the result, we use plot to emit the test data with the prediction
for idx, (class_id, image) in enumerate(zip(result, image_list)):
    plt.subplot(5, 2, idx+1)
    plt.title(labels[class_id])
    plt.imshow(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
    plt.axis('off')

plt.show()
