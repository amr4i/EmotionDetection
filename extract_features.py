import os
import sys
import cv2
import numpy as np
import VGG.vgg16 as vgg 
from lbp.LocalBinaryPatterns import LocalBinaryPatterns

def get_vgg_object_features(img, path_to_weights):
	vggDesc = vgg.predict(img, path_to_weights)
	return vggDesc

def get_lbp(image, numPoints=57, radius=4):
	lbpDesc = LocalBinaryPatterns(numPoints, radius)
	hist = lbpDesc.describe(image)
	return hist
	
def extract_feature_vector(imagepath, VGG_weights):
	image = cv2.imread(imagepath)
	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	LBP_Desc = get_lbp(grayImage)
	VGG_Desc = get_vgg_object_features(image, VGG_weights)
	# print "-------------> LBP Descriptor: \n", LBP_Desc
	# print "-------------> VGG Descriptor: \n", VGG_Desc
	
	# TO BE FINALIZED
	feature_vector = np.concatenate((LBP_Desc, VGG_Desc))
	print np.size(feature_vector)
	return feature_vector


""" 
training_file: a document where each line contains the name of the image file, and its corresponding A-V value, separated by a comma. 
directory: the folder that contains all the image files.
"""
def get_features(training_file ,directory, VGG_weights_file):
	features = []
	labels = []
	with open(training_file, 'r') as trainFile:
		files = trainFile.readlines()
		for file in files:
			filename, file_label = file.split(",")
			filepath = os.path.join(directory, filename)
			feature_vector = extract_feature_vector(filepath, VGG_weights_file)
			features.append(feature_vector)
			labels.append(file_label)
		return features, labels
	
if __name__ == "__main__":
	imagepath = sys.argv[1]
	vgg_weights = sys.argv[2]
	extract_feature_vector(imagepath, vgg_weights)

