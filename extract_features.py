import os
import sys
import cv2
import numpy as np
import VGG.vgg16 as vgg 
from lbp.LocalBinaryPatterns import LocalBinaryPatterns


names=[]
gists=[]
sem_features = {}

def load_gistfile(gistFile):
	with open(gistFile, 'r') as g:
		gs = g.readlines()
		for line in gs:
			name, gist = line.split(":")
			names.append(name)
			gists.append(gist)
	
def load_sematic_features(file):
	with open(file, 'rb') as handle:
		sem_features = pickle.load(handle)

def get_vgg_object_features(img, path_to_weights):
	vggDesc = vgg.predict(img, path_to_weights)
	return vggDesc

def get_lbp(image, numPoints=57, radius=4):
	lbpDesc = LocalBinaryPatterns(numPoints, radius)
	hist = lbpDesc.describe(image)
	return hist

def get_gist(imagename):
	ind = names.index(imagename)
	gist = gists[ind].strip().split(" ")
	for i in range(0,len(gist)):
		gist[i] = float(gist[i])
	gist = np.array(gist)
	return gist

def extract_feature_vector(imagepath, VGG_weights):
	imagename = imagepath.split("/")[-1]
	image = cv2.imread(imagepath)
	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	LBP_Desc = get_lbp(grayImage)
	GIST_Desc = get_gist(imagename)
	VGG_Desc = get_vgg_object_features(image, VGG_weights)
	Sem_Desc = sem_features[imagename]
	# print "-------------> LBP Descriptor: \n", LBP_Desc
	# print "-------------> VGG Descriptor: \n", VGG_Desc
	
	# TO BE FINALIZED
	feature_vector = np.concatenate((LBP_Desc, VGG_Desc, GIST_Desc, Sem_Desc))
	# print np.size(feature_vector)
	return feature_vector




""" 
training_file: a document where each line contains the name of the image file, and its corresponding A-V value, separated by a comma. 
directory: path to folder that contains all the image files.
"""
def get_features(training_file ,directory, VGG_weights_file, gistFile="gists.txt", semF_file='sematic_feature.pickle'):
	features = []
	labels = []
	
	with open(training_file, 'r') as trainFile:
		
		files = trainFile.readlines()
		load_gistfile(gistFile)
		load_semantic_features(semF_file)

		for file in files:
			filename, file_label = file.split(",")
			filepath = os.path.join(directory, filename)
			if not os.path.isabs(filepath):
				filepath = os.path.join(os.path.dirname(__file__), filepath)
			feature_vector = extract_feature_vector(filename, filepath, VGG_weights_file)
			features.append(feature_vector)
			labels.append(file_label)
		
		return features, labels
	
if __name__ == "__main__":
	imagepath = sys.argv[1]
	vgg_weights = sys.argv[2]
	load_gistfile(sys.argv[3])
	extract_feature_vector(imagepath, vgg_weights)

