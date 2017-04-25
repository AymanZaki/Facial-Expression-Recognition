import csv
import numpy as np
import tensorflow as tf
from PIL import Image
from numpy import array
from scipy.misc import toimage
from scipy.misc import imresize
from resizeimage import resizeimage


class FER2013_Input_Keras:
	Training_labels = []
	Training_Images = []
	Validation_labels = []
	Validation_Images = []
	Testing_labels = []
	Testing_Images = []
	Batch_Size = 128
	path = ""
	scale = 0
	Scale_1_Length = 42
	Scale_1_Width = 42
	Scale_2_Length = 84
	Scale_2_Width = 84
	Scale_3_Length = 90
	Scale_3_Width = 90
	def __init__(self, path, scale):
		self.path = path
		self.scale = scale
		"""[self.Training_labels, self.Training_Images] = self.FER2013_Training_Set()
		[self.Validation_labels, self.Validation_Images] = self.FER2013_Validation_Set()
		[self.Testing_labels, self.Testing_Images] = self.FER2013_Testing_Set()"""
		
	def FER2013_Training_Set (self):
			path = self.path + 'FER2013-Training.csv'
			Training_labels = []
			Training_Images = []
			with open(path) as csvfile:
				readCSV = csv.reader(csvfile, delimiter = ',')
				#row[0] = Labels, row[1] = Image Pixels
				for row in readCSV:					
					Label, ResizedImage = self.Get_Label_Image(row)
					Training_labels.append(Label)
					Training_Images.append(ResizedImage)					
			return  Training_labels, np.array(Training_Images)


	def FER2013_Validation_Set (self):
			Validation_labels = []
			Validation_Images = []
			path = self.path + 'FER2013-Validation.csv'
			with open(path) as csvfile:
				readCSV = csv.reader(csvfile, delimiter = ',')
				#row[0] = Labels, row[1] = Image Pixels
				for row in readCSV:
					Label, ResizedImage = self.Get_Label_Image(row)	
					Validation_labels.append(Label)
					Validation_Images.append(ResizedImage)
			return  Validation_labels, np.array(Validation_Images)

	def FER2013_Testing_Set (self):
			Testing_labels = []
			Testing_Images = []
			path = self.path + 'FER2013-Testing.csv'
			with open(path) as csvfile:
				readCSV = csv.reader(csvfile, delimiter = ',')	
				#row[0] = Labels, row[1] = Image Pixels
				for row in readCSV:
					Label, ResizedImage = self.Get_Label_Image(row)
					Testing_labels.append(Label)
					Testing_Images.append(ResizedImage)
			return  Testing_labels, np.array(Testing_Images)

	def Get_Label_Image(self, row):
			Label = int(row[0])
			image = row[1].split()
			#Cast strings to integer 
			image = [int(i) for i in image]
			#Convert 1D array to 2D array 48x48  
			Image48x48 = np.reshape(image, (48, 48))
			#Convert 2D array to Image
			ResizedImage = Image.fromarray(np.uint8(Image48x48))
			#Resize Image
			if(self.scale == 1):
				ResizedImage = imresize(ResizedImage, [self.Scale_1_Length, self.Scale_1_Width], interp='bilinear', mode=None)
				#ResizedImage = resizeimage.resize_contain(ResizedImage, [self.Scale_1_Length, self.Scale_1_Width])
			elif(self.scale == 2):
				ResizedImage = imresize(ResizedImage, [self.Scale_2_Length, self.Scale_2_Width], interp='bilinear', mode=None)
				#ResizedImage = resizeimage.resize_contain(ResizedImage, [self.Scale_2_Length, self.Scale_2_Width])
			elif(self.scale == 3):
				ResizedImage = imresize(ResizedImage, [self.Scale_3_Length, self.Scale_3_Width], interp='bilinear', mode=None)
			return Label, ResizedImage
"""
	#Batch_Number 0 based
	def Get_batch(self, Batch_Number, Mode):
		begin = Batch_Number * self.Batch_Size
		end = begin + self.Batch_Size
		Labels = []
		Images = []
		if(Mode == 'Training'):
			end = min(end, len(self.Training_labels) + 1)
			Labels = self.Training_labels[begin:end]
			Images = self.Training_Images[begin:end]
		elif (Mode == 'Validation'):
			end = min(end, len(self.Validation_labels) + 1)
			Labels = self.Validation_labels[begin:end]
			Images = self.Validation_Images[begin:end]
		elif (Mode == 'Testing'):
			end = min(end, len(self.Testing_labels) + 1)
			Labels = self.Testing_labels[begin:end]
			Images = self.Testing_Images[begin:end]
		return Labels, Images
		"""
