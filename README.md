# Yolo dataset converter

## Description
#####  _Convert your image dataset to yolo format!!_
This is a simple script to prepare your image dataset for object detection with YOLO models. 


## Usage
```python
from yolo_converter import YoloConverter
img_set = "images/"
ds_names = ["animal_set"]
classes = [["cat", "dog", "others"]]
train_idx = (300, 2500)

yc = YOLOConverter(img_set=img_set,dataset_names=ds_names, classes=classes, train_idx=train_idx)
yc.from_csv(train_csv="train.csv", test_csv= "test.csv", dataset_name="animal_set")
```
## Features
-  YOLO directories hierarchy
-  Multiple dataset allowed
