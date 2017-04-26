import os
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras import optimizers
from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
import h5py
from sklearn.cross_validation import train_test_split
import cv2
import csv
from helper import *
#K.set_image_dim_ordering('th')
print("packages imported")


PATH = "./train/"
train_path = "new_train/"
val_path = "new_val/"
img_size = (224, 224) #width, length
num_class = 3
num_epoch_1 = 3
num_epoch_2 = 100
X_0 = []
X_1 = []
X_2 = []
y_0 = []
y_1 = []
y_2 = []
X_train = []
y_train = []
X_val = []
y_val = []

data_0 = pd.read_csv('labels/labels_0.csv')
data_1 = pd.read_csv('labels/labels_1.csv')
data_2 = pd.read_csv('labels/labels_2.csv')

num_data_0 = data_0.shape[0]
num_data_1 = data_1.shape[0]
num_data_2 = data_2.shape[0]


def image_generator(image):
    image = augment_brightness(image)
    image = translation(image)
    image = flip(image)
    image = rotation(image)
    #image = shadow_mask(image)
    image = random_shear(image)
    return image



for i in tqdm(range(num_data_0)):
    imagename = data_0.iloc[i][0]
    image_path = PATH+imagename
    img = plt.imread(image_path)
    #img = np.asarray(img)
    #x = image.img_to_array(img)
    X_0.append(img)
    y_0.append(data_0.iloc[i][1])
#X_0 = np.asarray(X_0)
#X_0 = preprocess_input(X_train)
y_0 = np.asarray(y_0)


X_0_train, X_0_validation, y_0_train, y_0_validation = train_test_split(
    X_0, y_0, test_size=0.25, random_state=42)

#print("X_0_train size:", X_0_train.shape[0])
#print("X_0_test size:", X_0_test.shape[0])


for i in tqdm(range(num_data_1)):
    imagename = data_1.iloc[i][0]
    image_path = PATH+imagename
    img = plt.imread(image_path)
    #img = np.asarray(img)
    #x = image.img_to_array(img)
    X_1.append(img)
    y_1.append(data_1.iloc[i][1])
#X_1 = np.asarray(X_1)
#X_1 = preprocess_input(X_train)
y_1 = np.asarray(y_1)


X_1_train, X_1_validation, y_1_train, y_1_validation = train_test_split(
    X_1, y_1, test_size=0.0917, random_state=42)

#print("X_1_train size:", X_1_train.shape[0])
#print("X_1_test size:", X_1_test.shape[0])


for i in tqdm(range(num_data_2)):
    imagename = data_2.iloc[i][0]
    image_path = PATH+imagename
    img = plt.imread(image_path)
    #img = np.asarray(img)
    #x = image.img_to_array(img)
    X_2.append(img)
    y_2.append(data_2.iloc[i][1])
#X_2 = np.asarray(X_2)
#X_2 = preprocess_input(X_train)
y_2 = np.asarray(y_2)


X_2_train, X_2_validation, y_2_train, y_2_validation = train_test_split(
    X_2, y_2, test_size=0.1708, random_state=42)

#print("X_2_train size:", X_2_train.shape[0])
#print("X_2_test size:", X_2_test.shape[0])


count = 0

for i in range(4):
    for j in range(len(X_0_train)):
        image = X_0_train[j]
        if i == 0:
            path = train_path + str(count) + ".jpg"
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, image)
            X_train.append(str(count) + ".jpg")
            y_train.append(y_0_train[j])
            count = count + 1
        else:
            image = image_generator(image)
            path = train_path + str(count) + ".jpg"
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, image)
            X_train.append(str(count) + ".jpg")
            y_train.append(y_0_train[j])
            count = count + 1

for j in range(len(X_1_train)):
    image = X_1_train[j]
    path = train_path + str(count) + ".jpg"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, image)
    X_train.append(str(count) + ".jpg")
    y_train.append(y_1_train[j])
    count = count + 1

for i in range(2):
    for j in range(len(X_2_train)):
        image = X_2_train[j]
        if i == 0:
            path = train_path + str(count) + ".jpg"
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, image)
            X_train.append(str(count) + ".jpg")
            y_train.append(y_2_train[j])
            count = count + 1
        else:
            image = image_generator(image)
            path = train_path + str(count) + ".jpg"
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, image)
            X_train.append(str(count) + ".jpg")
            y_train.append(y_2_train[j])
            count = count + 1

df = pd.DataFrame({"image":X_train, "label":y_train})
df = df.sample(frac=1)
df.to_csv("large_train.csv", sep='\t')
print("large_train data size:", df.shape)



count = 50000
for j in range(len(X_0_validation)):
    image = X_0_validation[j]
    path = val_path + str(count) + ".jpg"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, image)
    X_val.append(str(count) + ".jpg")
    y_val.append(y_0_validation[j])
    count = count + 1
for j in range(len(X_1_validation)):
    image = X_1_validation[j]
    path = val_path + str(count) + ".jpg"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, image)
    X_val.append(str(count) + ".jpg")
    y_val.append(y_1_validation[j])
    count = count + 1
for j in range(len(X_2_validation)):
    image = X_2_validation[j]
    path = val_path + str(count) + ".jpg"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, image)
    X_val.append(str(count) + ".jpg")
    y_val.append(y_2_validation[j])
    count = count + 1

df = pd.DataFrame({"image":X_val, "label":y_val})
df = df.sample(frac=1)
df.to_csv("large_val.csv", sep='\t')
print("large_val data size:", df.shape)


