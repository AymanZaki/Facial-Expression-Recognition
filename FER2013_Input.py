import csv
import numpy

import numpy as np

class FER2013_Input:
    	
	path = ""
	def __init__(self, path):
		self.path = path
	
	def FER2013_Training_Set (self):
			Training_labels = []
			Training_Images = []
			path = self.path + 'FER2013-Training.csv'
			with open(path) as csvfile:
				readCSV = csv.reader(csvfile, delimiter = ',')
				for row in readCSV:
					Training_labels.append(row[0])
					Image48x48 = np.reshape(row[1].split(), (48, 48))
					Training_Images.append(Image48x48)
			return  Training_labels, Training_Images


	def FER2013_Validation_Set (self):
			Validation_labels = []
			Validation_Images = []
			path = self.path + 'FER2013-Validation.csv'
			with open(path) as csvfile:
				readCSV = csv.reader(csvfile, delimiter = ',')
				for row in readCSV:
					Validation_labels.append(row[0])
					Image48x48 = np.reshape(row[1].split(), (48, 48))
					Validation_Images.append(Image48x48)
			return  Validation_labels, Validation_Images

	def FER2013_Testing_Set (self):
			Testing_labels = []
			Testing_Images = []
			path = self.path + 'FER2013-Testing.csv'
			with open(path) as csvfile:
				readCSV = csv.reader(csvfile, delimiter = ',')			
				for row in readCSV:
					Testing_labels.append(row[0])
					Image48x48 = np.reshape(row[1].split(), (48, 48))
					Testing_Images.append(Image48x48)
			return  Testing_labels, Testing_Images