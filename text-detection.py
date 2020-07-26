import os
import re
import cv2 as cv 
import numpy as np 
from PIL import Image
import imutils
import pytesseract as tess 


def plateExtract(path):
	img = np.array(Image.open(path).convert('RGB'))
	height, width = img.shape[:2]

	gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
	gray = cv.bilateralFilter(gray, 11, 17, 17)
	edgeMap = cv.Canny(gray, 180, 200)

	contours, hierarchy = cv.findContours(edgeMap, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	image = img.copy()
	contours.sort(key=cv.contourArea, reverse=True)

	for cnt in contours:
		approx = cv.approxPolyDP(cnt, cv.arcLength(cnt, True)/50, True)

		if len(approx) == 4:
			x, y, w, h = cv.boundingRect(cnt)
			cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
			if w*h > 100:
				return textRecog(img[y:y+h, x:x+w, :])
			
	return textRecog(img) 


def textRecog(img):
	height, width = img.shape[:2]
	data = tess.image_to_boxes(img).split('\n')

	if data == ['']:
		return '', img

	boxes = []
	text = []

	for line in data:
		row = line.split()
		boxes.append([row[0], [int(x) for x in row[1:]]])
		text.append(row[0])

	for box in boxes:
		x, y, w, h = box[1][0], box[1][1], box[1][2], box[1][3] 
		img = cv.rectangle(img, (x, height-y), (w, height-h), (36,255,12), 1)
		cv.putText(img, box[0], (x, height-h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (12,20,255), 2)

	return ''.join(text), img 


if __name__ == '__main__':
	tess.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

	ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
	TRAIN_DIR = os.path.join(ROOT_DIR, 'test')
	RESULT_DIR = os.path.join(ROOT_DIR, 'result')	
	
	inp = input()
	
	if inp == '__all__':
		for img in os.listdir(TRAIN_DIR):
			path = os.path.join(TRAIN_DIR, img)

			text, image = plateExtract(path)
			image = imutils.resize(image, width=500)		
			cv.imwrite(os.path.join(RESULT_DIR, img), image)
			
			txt = ''.join(img.split('.')[:-1]) + '.txt' 

			with open(os.path.join(RESULT_DIR, txt), 'w') as f:
				
				if re.match(r'^[A-Z]{2}\d+[A-Z]{2}\d+$', text):
					text += '\nValid License Plate'
				else:
					text += '\nInvalid License Plate'
				
				f.write(text)

	else:
		img = inp 
		path = os.path.join(TRAIN_DIR, img)

		text, image = plateExtract(path)
		image = imutils.resize(image, width=500)
		cv.imwrite(os.path.join(RESULT_DIR, img), image)
		
		txt = ''.join(img.split('.')[:-1]) + '.txt' 
		with open(os.path.join(RESULT_DIR, txt), 'w') as f:

			if re.match(r'^[A-Z]{2}\d+[A-Z]{2}\d+$', text):
				text += '\nValid License Plate'
			else:
				text += '\nInvalid License Plate'

			f.write(text)
