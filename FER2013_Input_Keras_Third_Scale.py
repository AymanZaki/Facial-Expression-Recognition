import csv
import numpy as np
import tensorflow as tf
from PIL import Image
from numpy import array
from scipy.misc import toimage
from resizeimage import resizeimage


class FER2013_Input_Keras_Third_Scale:
	Training_labels = []
	Training_Images = []
	Validation_labels = []
	Validation_Images = []
	Testing_labels = []
	Testing_Images = []
	Batch_Size = 128
	path = ""
	def __init__(self, path):
		self.path = path
		[self.Training_labels, self.Training_Images] = self.FER2013_Training_Set()
		[self.Validation_labels, self.Validation_Images] = self.FER2013_Validation_Set()
		[self.Testing_labels, self.Testing_Images] = self.FER2013_Testing_Set()
		
	def FER2013_Training_Set (self):
			path = self.path + 'FER2013-Training.csv'
			Training_labels = []
			Training_Images = []
			with open(path) as csvfile:
				readCSV = csv.reader(csvfile, delimiter = ',')
				#row[0] = Labels, row[1] = Image Pixels
				for row in readCSV:
					Label = int(row[0])
					#Label = [0] * 7
					#Label[int(row[0])] = 1
					Training_labels.append(Label)
					image = row[1].split()
					#Cast strings to integer 
					image = [int(i) for i in image]
					#Convert 1D array to 2D array 48x48
					Image48x48 = np.reshape(image, (48, 48))
					#Convert 2D array to Image
					Image90x90 = Image.fromarray(np.uint8(Image48x48))
					#Resize Image to 90x90
					Image90x90 = resizeimage.resize_contain(Image90x90, [90, 90])
					#Convert Image to 2D array 
					Image90x90 = np.uint8(Image90x90.convert('L'))
					Training_Images.append(Image90x90)
			return  Training_labels, np.array(Training_Images)


	def FER2013_Validation_Set (self):
			Validation_labels = []
			Validation_Images = []
			path = self.path + 'FER2013-Validation.csv'
			with open(path) as csvfile:
				readCSV = csv.reader(csvfile, delimiter = ',')
				#row[0] = Labels, row[1] = Image Pixels
				for row in readCSV:
					Label = int(row[0])
					#Label = [0] * 7
					#Label[int(row[0])] = 1
					Validation_labels.append(Label)
					image = row[1].split()
					#Cast strings to integer 
					image = [int(i) for i in image]
					#Convert 1D array to 2D array 48x48
					Image48x48 = np.reshape(image, (48, 48))
					#Convert 2D array to Image
					Image90x90 = Image.fromarray(np.uint8(Image48x48))
					#Resize Image to 90x90
					Image90x90 = resizeimage.resize_contain(Image90x90, [90, 90])
					#Convert Image to 2D array 
					Image90x90 = np.uint8(Image90x90.convert('L'))		
					Validation_Images.append(Image90x90)
			return  Validation_labels, np.array(Validation_Images)

	def FER2013_Testing_Set (self):
			Testing_labels = []
			Testing_Images = []
			path = self.path + 'FER2013-Testing.csv'
			with open(path) as csvfile:
				readCSV = csv.reader(csvfile, delimiter = ',')	
				#row[0] = Labels, row[1] = Image Pixels
				c = 0
				for row in readCSV:
					Label = int(row[0])
					#Label = [0] * 7
					#Label[int(row[0])] = 1
					Testing_labels.append(Label)
					image = row[1].split()
					#Cast strings to integer 
					image = [int(i) for i in image]
					#Convert 1D array to 2D array 48x48  
					Image48x48 = np.reshape(image, (48, 48))
					#Convert 2D array to Image
					Image90x90 = Image.fromarray(np.uint8(Image48x48))
					#Resize Image to 90x90
					Image90x90 = resizeimage.resize_contain(Image90x90, [90, 90])
					#Convert Image to 2D array 
					Image90x90 = np.uint8(Image90x90.convert('L'))
					Testing_Images.append(Image90x90)
					c = c + 1
					#if c>=400:
					#	break
			return  Testing_labels, np.array(Testing_Images)

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
