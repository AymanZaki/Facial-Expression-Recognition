import csv
import numpy as np
import tensorflow as tf
from PIL import Image

class FER2013_Input:
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
				for row in readCSV:
					Training_labels.append(int(row[0]))
					Image = row[1].split()
					Image = [int(i) for i in Image]
					Image48x48 = np.reshape(Image, (48, 48))
					Training_Images.append(Image48x48)
			return  Training_labels, Training_Images


	def FER2013_Validation_Set (self):
			Validation_labels = []
			Validation_Images = []
			path = self.path + 'FER2013-Validation.csv'
			with open(path) as csvfile:
				readCSV = csv.reader(csvfile, delimiter = ',')
				for row in readCSV:
					Validation_labels.append(int(row[0]))
					Image = row[1].split()
					Image = [int(i) for i in Image]
					Image48x48 = np.reshape(Image, (48, 48))
					Validation_Images.append(Image48x48)
			return  Validation_labels, Validation_Images

	def FER2013_Testing_Set (self):
			Testing_labels = []
			Testing_Images = []
			path = self.path + 'FER2013-Testing.csv'
			with open(path) as csvfile:
				readCSV = csv.reader(csvfile, delimiter = ',')			
				for row in readCSV:
					Testing_labels.append(int(row[0]))
					Image = row[1].split()
					Image = [int(i) for i in Image]
					Image48x48 = np.reshape(Image, (48, 48))
					Testing_Images.append(Image48x48)
			return  Testing_labels, Testing_Images

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