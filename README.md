## EmotionDetection

1. Make sure openCV is installed on both python and C++. 
Installing OpenCV on C++: 
http://www.codebind.com/cpp-tutorial/install-opencv-ubuntu-cpp/

2. make gist executable by 
```g++ -std=c++11 -o gist main.cpp $(pkg-config opencv --cflags --libs)```

3. Give the absolute path to the images folder. 

```./gist -i <absolute path to folder containing images> -o <directory where output files are needed> ```


VGG Descriptor:
http://www.cs.toronto.edu/~frossard/post/vgg16/ 



LBP Descriptor: 
https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/