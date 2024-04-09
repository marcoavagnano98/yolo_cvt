# Yolo dataset converter

## Description
#####  _Convert your image dataset to yolo format!!_
This is a simple script to prepare your image dataset for object detection with YOLO models. 
This script take as input a train and test csv files formed in this way:
```csv
image_id,bbox,category_id
id1,"[122.0, 1.0, 42.0, 30.0]",2.0
id2,"[122.0, 1.0, 42.0, 30.0]",1.0
id3,"[122.0, 1.0, 42.0, 30.0]",3.0
```
Coordinates of a bounding box are encoded with four values in pixels: [x_min, y_min, width, height].
It's possible to specify more bounding boxes with same ```image_id```  then the script will convert the bounding box coordinates to YOLO format and will produce one txt file per image. 
The images in the specified folder will be organized in the YOLO directories hierarchy and divided in the train and validation set according to train indices.
The ```category_id``` represent the object class and it can be either a digit or a class name string.

## Usage
```python
from yolo_converter import YoloConverter
img_set = "images/"
ds_names = ["animal_set"]
categories = [{"cat":1, "dog":2, "others":3}] # or [["cat", "dog", "others"]]
train_idx = (300, 2500)

yc = YOLOConverter(img_set=img_set,dataset_names=ds_names, classes=categories, train_idx=train_idx)
yc.from_csv(train_csv="train.csv", test_csv= "test.csv", dataset_name="animal_set")
```
Submitted categories can be either ``` dict ``` or ``` list ```, in the first case the csv file must contain ```category_id``` of type string (category name).
## Features
-  YOLO directories hierarchy
-  Multiple dataset allowed
-  Multi-type categories
