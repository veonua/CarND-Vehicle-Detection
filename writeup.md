**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/heatmap.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in in `get_HOG_features()` in utils.py).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I have explored different HOG features that we could extract and test.  After each run, the result of the model was saved as well as its scaled_X transform with `scikit-learn` `joblib.dump()` function.  We will use the best saved model for our final vehicle detection implementation in our pipeline.  The following table shows the HOG features I explored:

| HOG Name | Color Space | HOG Channel | Orientation | Pixel/Cell | Cell/Block | Jittered | Train Accuracy | Test Accuracy | Prediction Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CHOGLUV1 | LUV | 1 | 9 | 8 | 2 | No | 1.000 | 0.975 | 0.000141 secs |
| CHOGHLS1 | HLS | 2 | 9 | 8 | 2 | No | 1.000 | 0.969 | 0.000142 secs |
| CHOGYUV1 | YUV | 0 | 9 | 8 | 2 | No | 1.000 | 0.985 | **0.000102 secs** |
| CHOGYUV2 | YUV | 0 | 9 | 4 | 2 | No | 1.000 | **0.987** | 0.000141 secs |
| CHOGGRAY1 | Grayscale | Grayscale | 9 | 8 | 2 | No | 1.000 | 0.956 | 0.000144 secs |
| CHOGGRAYRGB1 | Both Grayscale and RGB | Grayscale | 9 | 4 | 2 | No | 1.000 | 0.986 | 0.00035 secs |
| CHOGHLS2 | HLS | 2 | 9 | 4 | 2 | No  | 1.000 | 0.967 | 0.000145 secs |
| CHOGRGB1 | RGB | 0 | 9 | 8 | 2 | No  | 1.000 | *0.986* | 0.000142 secs |
| CHOGRGB2 | RGB | 0 | 8 | 4 | 2 | No  | 1.000 | 0.976 | 0.000142 secs |
| CHOGRGB3 | RGB | 0 | 9 | 8 | 2 | Yes | 0.968 | 0.946 | 0.000144 secs
| CHOGRGB4 | RGB | 0 | 8 | 4 | 2 | Yes | 0.999 | 0.948 | 0.000202 secs |
| CHOGHSV1 | HSV | 1 | 9 | 8 | 2 | No  | 1.000 | 0.980 | 0.000140 secs |
| CHOGRGB5 | RGB | 0 | 9 | 2 | 2 | Yes | 1.000 | 0.917 | 0.000170 secs |


Then I applied the binned color features and the histograms of color in the color space to the HOG feature vector, except for the CHOGGRAY1, which only has the Histogram of Oriented Gradients vectors.  After forming the feature vector, it is then normalized using the `Sci-Kit Learn` `StandardScaler` function to normalize the vector before training with the SVM linear classifier.

The HOG feature set for CHOGYUV2 seems to have the best accuracy, 0.987; follow closely by CHOGRGB1 with 0.986 accuracy.  CHOGYUV1 had the best timing with just 0.000102 seconds to do a prediction, but with a lower accuracy of 0.985. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG, color bin and color histogram features in train_model() at line 668 of utils.py. To save on iteration time, I saved the trained models so I could load them without retraining. My final model was a Linear SVM trained on YUV images, the aforementioned HOG settings, color bin features of shape (16, 16) and color histogram features with 32 bins. This resulted in a test accuracy of 98.6%. I trained on the GTI data and tested on the other directories to avoid the risk of overfitting that came with having similar images in the same directory.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I searched on windows of scales 64x64 and 96x96 over the whole road area (y pixels 400-656), because these searches are relatively quick and produce good results. I also searched at scale 32x32, limited to y pixels 400-500, because this search took much longer, and smaller car detections are more likely to be closer to the horizon. Here is an example of a heatmap of detections made on a test image:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

### Optimization

Unfortunately sliding window approach is very slow due to inefficient overlapping and it is not optimized for run on GPUs. It took over 5 seconds to process one frome, which is far from realtime. So I tried Faster R-CNN method from [CS231n course](http://cs231n.stanford.edu/slides/2016/winter1516_lecture8.pdf). the TensorFlow implementation could be found at [Github](https://github.com/endernewton/tf-faster-rcnn)

Faster R-CNN was selected as it utilize resources more efficiently that speedup the processing almost 8x to ~0.7 seconds per frame using GPU. As I had limited amount of time I used pretrained 
ResNet with 101 layers, trained on Pascal VOC 2012 dataset


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/inWlBx4XBNo)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Faster R-CNN produces small amount of false positives. I choose to left Score threshold at 80%. Non-maximum suppression [NMS](./lib/nms/) combines overlapping bounding boxes. 

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I studied several approaches during this project, with pros and cons.
SVM classifier to HOG features covered it the Project lessons is simpler to implement but gives poor perfomance, also it is less agile than Deformable Part Models which is evolution of HOG idea.

Convolutions is the modern way to solve detection, Furthermore, there is a [paper](https://arxiv.org/abs/1409.5403) which argues that the DPMs (those are based on HOGs) might be considered as a certain type of Convolutional Neural Networks. 
Faster R-CNN is the evolution of R-CNN and Fast R-CNN archritectures, that solves detection with CNN on the whole image and for the last layers there are Region Proposal Network(RPN) and Region of Interest Pooling (RoI). So it does not need for external region proposals. And can be trained much faster.

Also I researched dense flow approach, to pay attention only to regions(blobs) that were changed 
https://people.eecs.berkeley.edu/~pathak/unsupervised_video/ but provided pyflow dense flow algorithm failed on given video. But I still belive unsupervised video learning is promising direction of research for these kind of problems.