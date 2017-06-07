from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json

from FER2013_Input_Keras import FER2013_Input_Keras
import csv
import numpy as np
import tensorflow as tf
from PIL import Image
from numpy import array
from scipy.misc import toimage
from resizeimage import resizeimage
from scipy.misc import toimage

import Face_Detection_Scale1
import Face_Detection_Scale2
import Face_Detection_Scale3
import dlib
import math


batch_size = 1
num_classes = 7
epochs = 1000

detect = Face_Detection_Scale1.Preprocessing()
Input_Images_Original = detect.Faces_Detection()
out = Input_Images_Original
img_rows, img_cols = 42,42

#fer = FER2013_Input_Keras('/home/alaa/Desktop/GP/')
#Testing_labels, Testing_Images = fer.FER2013_Testing_Set()
#Testing_Images = Testing_Images[0]
Input_Images = Input_Images_Original[:]
Input_Images = Input_Images.reshape(Input_Images.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

Input_Images = Input_Images.astype('float32')
Input_Images /= 255



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('model_weights.h5')

# evaluate loaded model on test data
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
score1 = loaded_model.predict(Input_Images, verbose=0)

detect = Face_Detection_Scale2.Preprocessing()
Input_Images = detect.Faces_Detection()
out = Input_Images
img_rows, img_cols = 84,84

#fer = FER2013_Input_Keras('/home/alaa/Desktop/GP/')
#Testing_labels, Testing_Images = fer.FER2013_Testing_Set()
#Testing_Images = Testing_Images[0]
Input_Images = Input_Images.reshape(Input_Images.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

Input_Images = Input_Images.astype('float32')
Input_Images /= 255



# load json and create model
json_file = open('modelSecondScale.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('modelSecondScale_weights.h5')

# evaluate loaded model on test data
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
score2 = loaded_model.predict(Input_Images, verbose=0)

detect = Face_Detection_Scale3.Preprocessing()
Input_Images = detect.Faces_Detection()
out = Input_Images
img_rows, img_cols = 90,90

#fer = FER2013_Input_Keras('/home/alaa/Desktop/GP/')
#Testing_labels, Testing_Images = fer.FER2013_Testing_Set()
#Testing_Images = Testing_Images[0]
Input_Images = Input_Images.reshape(Input_Images.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

Input_Images = Input_Images.astype('float32')
Input_Images /= 255



# load json and create model
json_file = open('modelThirdScale1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('modelThirdScale1_weights.h5')

# evaluate loaded model on test data
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
score3 = loaded_model.predict(Input_Images, verbose=0)

score = []
for i in range(len(score1)):
	temp = []
	for j in range(len(score1[i])):
		temp.append((score1[i][j]+score2[i][j]+score3[i][j])/3)
	score.append(temp)

'''score = loaded_model.predict_classes(Input_Images, verbose=0)
classes = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
for i in range(len(Input_Images)):
  toimage(out[i]).show()
  print(classes[score[i]])
  dlib.hit_enter_to_continue()
#print(score)
'''






classes = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

score_exp = []
softmax = []
output = []
for i in range(len(Input_Images)):
	score_exp.append([math.exp(j) for j in score[i]])
	sum_exp = sum(score_exp[i])
	softmax.append([round(j / sum_exp, 3) for j in score_exp[i]])
	index = 0
	maxVal = 0
	for j in range(len(softmax[i])):
		if(softmax[i][j]>=maxVal):
			index = j
			maxVal = softmax[i][j]
	output.append(classes[index])


print(output)