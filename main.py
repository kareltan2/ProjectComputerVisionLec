from email.mime import image
from unittest import result
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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
sift = cv2.SIFT_create()
des_list = []

for image in image_list:
    # Kp akan menunjukkan koordinat yang unique, sehingga kita tidak membutuhkan kp
    _, des = sift.detectAndCompute(cv2.imread(image), None)
    des_list.append(des)

# Preprocessing
stacked_des = des_list[0]

# Stack semua dari index 1
for des in des_list[1:]:
    # stack secara keatas
    stacked_des = np.vstack((stacked_des, des))

stacked_des = np.float32(stacked_des)

# Clustering
# kmeans
# centroid, distorsi = kmeans(stacked_des, 3, 20)
centroid, _ = kmeans(stacked_des, 3, 20)

image_feature = np.zeros((len(image_list), len(centroid)), dtype= "float32")

# Vector Quantization
# bikin histogram
for i in range(len(image_list)):
    words, _ = vq(des_list[i], centroid)

    for w in words:
        image_feature[i][w] += 1
    # Nilai nya bear maka harus melakukan standarisasi

std_scaler = StandardScaler().fit(image_feature)

# Kita tampung kembali yag sudah di tranform
image_feature = std_scaler.transform(image_feature)

# Classifiying
svc = LinearSVC()

svc.fit(image_feature, np.array(image_classes_list))

# Test
test_path = 'Dataset/Test'
test_path_list = os.listdir(test_path)

image_list = []

for image in test_path_list:
    # Dataset/Test/1.jpeg
    image_list.append(test_path + '/' + image)

sift = cv2.SIFT_create()
des_list = []

for image in image_list:
    _, des = sift.detectAndCompute(cv2.imread(image), None)
    des_list.append(des)


# Testing
test_feature = np.zeros((len(image_list), len(centroid)), "float32")

for i in range(len(image_list)):
    words, _ = vq(des_list[i], centroid)

    for w in words:
        test_feature[i][w] += 1

test_feature = std_scaler.transform(test_feature)
result = svc.predict(test_feature)

# Show Result
for idx, (class_id, image) in enumerate(zip(result, image_list)):
    plt.subplot(5, 2, idx+1)
    plt.title(labels[class_id])
    plt.imshow(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
    plt.axis('off')

plt.show()
