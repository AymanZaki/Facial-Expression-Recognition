import sys
import dlib
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image
from resizeimage import resizeimage
from scipy.misc import toimage
from scipy.misc import imresize

Resize_Length_Scale1 = 42
Resize_Width_Scale1 = 42
Resize_Length_Scale2 = 84
Resize_Width_Scale2 = 84
Resize_Length_Scale3 = 90
Resize_Width_Scale3 = 90
Max_Length = 1000
Max_Width = 1000

class Preprocessing:

	def Get_Image_Dimensions(self, image):
		Dimensions = np.shape(image)
		Length = Dimensions[0]
		Width = Dimensions[1]
		if(len(Dimensions) > 2):
			Channel = Dimensions[2]
		else:
			Channel = 0
		return [Length, Width, Channel]
	
	def Resize_Image(self, image, Length, Width):
		Resized_Image = resize(image, (Length, Width), mode='reflect')		
		return Resized_Image

	def Rgb2Gray(self, image):
		Gray_Image = rgb2gray(image)
		return Gray_Image

	def Cast2Int(self, image, Length, Width):
		Gray_Image = []
		for i in range(Length):
			Gray_Image.append([])
			for j in range(Width):
				Gray_Image[i].append (np.uint8(image[i][j] * 255.0))
		Gray_Image = np.asarray(Gray_Image)
		return Gray_Image

	def Faces_Detection(self):
		for img in sys.argv[1:]:
			#Read image form a path
			image = io.imread(img)
			[Length, Width, Channel] = self.Get_Image_Dimensions(image)
			if(Channel > 0):
				Gray_Image = self.Rgb2Gray(image)
			else:
				Gray_Image = image
			if(Length * Width > Max_Length * Max_Width):
				Length = int(Length * 0.3)
				Width = int(Width * 0.3)
				Gray_Image = self.Resize_Image(Gray_Image, Length, Width)

			if(Channel > 0):
				Gray_Image = self.Cast2Int(Gray_Image, Length, Width)
			
			detector = dlib.get_frontal_face_detector()
			Faces = detector(Gray_Image, 2)
			Faces_Scale1 = []
			Faces_Scale2 = []
			Faces_Scale3 = []
			for Face in Faces:
				x1 = Face.top()
				y1 = Face.left()
				x2 = Face.bottom()
				y2 = Face.right()
				Cropped_Image = Gray_Image[x1:x2, y1:y2]
				Resized_Image_Scale1 = self.Resize_Image(Cropped_Image, Resize_Length_Scale1, Resize_Width_Scale1)
				Resized_Image_Scale2 = self.Resize_Image(Cropped_Image, Resize_Length_Scale2, Resize_Width_Scale2)
				Resized_Image_Scale3 = self.Resize_Image(Cropped_Image, Resize_Length_Scale3, Resize_Width_Scale3)
				Faces_Scale1.append(Resized_Image_Scale1)
				Faces_Scale2.append(Resized_Image_Scale2)
				Faces_Scale3.append(Resized_Image_Scale3)
				#toimage(Resized_Image_Scale1).show()
				#toimage(Resized_Image_Scale2).show()
				#toimage(Resized_Image_Scale3).show()
			print ("number of faces detected: ", len(Faces))	
			Faces_Scale1 = np.array(Faces_Scale1)
			Faces_Scale2 = np.array(Faces_Scale2)
			Faces_Scale3 = np.array(Faces_Scale3)
		return Faces_Scale1, Faces_Scale2, Faces_Scale3

#tmp = Preprocessing();
#Faces = tmp.Faces_Detection()
#for face in Faces:
#	toimage(face).show()