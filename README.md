# Synthetic Image Classification

## Description ##
This is a proof of concept implementation of a Category classifier which is trained on a set of synthetically created set of 15k images. The images use a varying count on straight lines within an image and this variation defines the complexity within the image.  

The categories as defined in the system are:  
Category 0: 1-10 lines  
Category 1: 11-20 lines  
Category 2: 20-30 lines  

e.g. 
[Category 0 samples](images/Category_0_samples.png)
[Category 1 samples](images/Category_1_samples.png)
[Category 2 samples](images/Category_2_samples.png)

The DL network is a simple image analysis system which is fed a labelled set of these images with the category, and it learns predicting the categories once ready though its inferences.  

The implementation has 3 key components:  
1 A Image generation mechanism for the data set. (Generates 256x256, 3 channel images)  
1 One data normalisation routine.   
2 One training module.  
3 One prediction module with some test data.  

Training process is required only once, unless the pattern action data file is changed, in which case it should be rerun.  

## Dependencies ##
OpenCV  
Tensorflow    
Matplotlib  
Numpy  

## Results ##
The current network configuration generates a validation accuracy of 90%+.