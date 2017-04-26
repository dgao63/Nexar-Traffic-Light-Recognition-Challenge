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
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
import h5py
import cv2
from helper import *
#K.set_image_dim_ordering('th')
print("packages imported")


fine_tuning_model_weights_path = 'try.h5'
train_data = pd.read_csv('large_train.csv')
validation_data = pd.read_csv('large_val.csv')
TRAIN_PATH = './new_train/'
VAL_PATH = './new_val/'

num_train = train_data.shape[0]
num_validation = validation_data.shape[0]
X_train = []
y_train = []
X_validation = []
y_validation = []
img_size = (224, 224) #width, length
num_class = 3
num_epoch_1 = 1
num_epoch_2 = 1
BATCH_SIZE = 64
num_batch = num_train/BATCH_SIZE
print("num_batch:",num_batch)


for i in tqdm(range(num_validation)):
    imagename = validation_data.iloc[i][0]
    image_path = VAL_PATH+imagename
    img = image.load_img(image_path, target_size=img_size)
    x = image.img_to_array(img)
    X_validation.append(x)
    y_validation.append(validation_data.iloc[i][1])
X_validation = np.asarray(X_validation)
X_validation = preprocess_input(X_validation)
y_validation = np.asarray(y_validation)
y_validation = to_categorical(y_validation, num_class)

print("X_validation size:", X_validation.shape)
print("y_validation size:", y_validation.shape)
print("y_validation is like:", y_validation[0])

def do_image(image):
    image = augment_brightness(image, threshold=0.6)
    image = translation(image, threshold=0.6)
    image = flip(image, threshold=0.6)
    image = rotation(image, threshold=0.6)
    #image = shadow_mask(image)
    image = random_shear(image, threshold=0.6)
    return image

def data_gen(BATCH_SIZE=64):
    offset = 0
    while 1:
        print("offset:", offset)
        X_batch = []
        y_batch = []
        if offset + BATCH_SIZE > num_train:
            offset = 0
        batch_data = train_data.iloc[offset:offset+BATCH_SIZE]
        for i in range(BATCH_SIZE):
            imagename = batch_data.iloc[i][0]
            image_path = TRAIN_PATH+imagename
            #img = image.load_img(image_path, target_size=img_size)
            #x = image.img_to_array(img)
            img = plt.imread(image_path)
            img = cv2.resize(img, img_size)
            img = do_image(img)
            x = image.img_to_array(img)
            X_batch.append(x)
            y_batch.append(batch_data.iloc[i][1])
        X_batch = np.asarray(X_batch)
        X_batch = preprocess_input(X_batch)
        y_batch = np.asarray(y_batch)
        y_batch = to_categorical(y_batch, num_class)
        offset = offset + BATCH_SIZE
        yield (X_batch, y_batch)



base_model = InceptionV3(include_top=False, weights='imagenet')

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

x = base_model.layers[151].output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dropout(0.5, name='fc1_dropout')(x)
x = Dense(500, activation='relu', name='fc2')(x)
x = Dropout(0.5, name='fc2_dropout')(x)
predictions = Dense(3, activation='softmax', name='fc3_softmax')(x)

model = Model(input=base_model.input, output=predictions)

#for i, layer in enumerate(model.layers):
#   print(i, layer.name)

for layer in base_model.layers:
    layer.trainable = False
sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
train_generator = data_gen(BATCH_SIZE=64)
model.fit_generator(generator=train_generator, 
    samples_per_epoch=num_batch, epochs=num_epoch_1, validation_data=(X_validation, y_validation))
y = model.predict(X_validation, batch_size=32)
y = y.tolist()
df = pd.DataFrame({"pred":y})
df = df.sample(frac=1)
df.to_csv("pred.csv", sep='\t')

#the top layers are trained, start fine-tuning
for layer in base_model.layers[:10]:
   layer.trainable = True
for layer in base_model.layers[10:]:
   layer.trainable = True

sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
train_generator = data_gen(BATCH_SIZE=64)
model.fit_generator(generator=train_generator, 
    samples_per_epoch=num_batch, epochs=num_epoch_2, validation_data=(X_validation, y_validation))
acc = model.evaluate(X_validation, y_validation, batch_size=32)
print(acc)
y = model.predict(X_validation, batch_size=32)
y = y.tolist()
df = pd.DataFrame({"pred":y})
df.to_csv("pred.csv", sep='\t')
model.save_weights(fine_tuning_model_weights_path)
