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

Resize_Length = 42
Resize_Width = 42
Max_Length = 1500
Max_Width = 1500

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
			Face_Pixels = []
			for Face in Faces:
				x1 = Face.top()
				y1 = Face.left()
				x2 = Face.bottom()
				y2 = Face.right()
				Cropped_Image = Gray_Image[x1:x2, y1:y2]
				Resized_Image = self.Resize_Image(Cropped_Image, Resize_Length, Resize_Width)
				Face_Pixels.append(Resized_Image)
			print ("number of faces detected: ", len(Faces))	
			
		return np.array(Face_Pixels)

#tmp = Preprocessing();
#Faces = tmp.Faces_Detection()
#for face in Faces:
#	toimage(face).show()