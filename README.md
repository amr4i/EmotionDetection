# EmotionDetection

This is a python implementation for extracting arousal and valence values from images, as presented in the work: 
[Building Emotional Machines: Recognizing Image Emotions through Deep Neural Networks](https://arxiv.org/pdf/1705.07543.pdf) by Hye-Rin Kim, Yeong-Seok Kim, Seon Joo Kim, In-Kwon Lee. 

##Pre-requisites

1. Make sure openCV is installed on both python and C++. 
Here is a [tutorial](http://www.codebind.com/cpp-tutorial/install-opencv-ubuntu-cpp/) to installing OpenCV on C++.

##Instructions

####Training

1. To avoid loading of model weights again and again, we first pre-compute certain features for all images, that will be used in making the feature vector of the image. 
	- GIST Feature Extraction: 
		- Run the code segment:  
``` 
			cd GIST 
			make IMAGEPATH=<path_to_directory_containing_all_images>
			make clean
```	
		- This will create a file in the main folder named **gists.txt**, that contains the GIST descriptor of each image, one per line, in the format:
			> <FILENAME>:<GIST_Descriptor>
	- VGG Object Feature Extraction:
		- Run the code segment:
```
		cd VGG
		python vgg16.py <imageFile> <imagesDir> <VGG_weights>
```
		where *imageFile* = a file containing names of images and their A/V values separated by a comma, and *imagesDir* = the folder containing all training images. 
		- This will create a pickle in the main folder named **vgg.pickle**, that holds a python dictionary with names of images being mapped to their VGG object features.
		- More information about this VGG Descriptor can be found [here](http://www.cs.toronto.edu/~frossard/post/vgg16/).
	- Semantic Features Extraction:
		- Run the file _test.py_ in the _semantic_features_  folder, providing 4 python arguments, namely: *test_img_file*, *test_img_directory*, *weights_encoder* and *weights_decoder*.   

2. Now, using these extracted features, and more ([LBP features](https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
)), we will be constructing a feature vector for each training image. 

3. Set the training parameters in the file _params.json_, as required. 

4. Run the training file:
```
	python train.py <imageFile> <imagesDir>
```

5. This will store the model in a *modelData* named directory, in the parent directory of the current folder. The name of folder in which contains a *timestamp* that will be used to recognise the stroed model during the prediction. 

####Prediction

1. Again, go through the entire process of feature exrraction for the images for which the prediction needs to be made. 

2. Run the prediction code:
```
	python predict.py <testImageFile> <testImgDir> <timestamp>
```
where *testImageFile*=file containing names of all images to be predicted for, *testImgDir*=directory containing all images to predicted for, *timestamp*=the timestamp of the model to be used for prediction. 