import pandas as pd
import cv2
import os
import shutil
import ast
from tqdm import tqdm

class YOLOConverter:
    def __init__(self, img_set, dataset_names, classes):
        self.img_set = img_set
        self.dataset_names = dataset_names
        self.category_map = {}
        self.data = {}

        assert isinstance(dataset_names, list), f"dataset_names must be a list type"

        if isinstance(img_set, list):
            assert len(img_set) == len(dataset_names) and len(classes) == len(img_set), f"The length of image set, dataset names and classes list must be equal"
        else:
            assert len(dataset_names) == 1 and len(classes) == 1, f"A single image dataset folder has been specified, and multiple dataset names or classes"
        
        self.classes = {name: c for name, c in zip(dataset_names, classes)}

    def __make_dir(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

    def __prepare_directories(self):
        parent_dir = "yolo"
        self.node_dir = f"{parent_dir}/datasets"
        self.datasets_path = [os.path.join(self.node_dir, name) for name in self.dataset_names]
        self.data = {name:p for p, name in zip(self.datasets_path, self.dataset_names)}
        image_subdirs = ["images", "images/labels", "images/labels/train", "images/labels/val", "images/train", "images/val"]
        for dataset_path in self.datasets_path:
            ds = [parent_dir, self.node_dir, dataset_path] + [os.path.join(dataset_path, subdir) for subdir in image_subdirs]
            for d in ds:
                self.__make_dir(d)
        
    def __get_yaml(self, dataset_path,name):
        out = f"path: ../{dataset_path}\ntrain: images/train\nval: images/val\n\nnames:\n"
        _class  = self.classes[name]
        for idx, c in enumerate(_class):
            cline = f"\t{idx}: {c}"
            out += cline + "\n"
        return out

    def __prepare_yaml(self):
        for p, n in zip(self.datasets_path, self.dataset_names):
            with open(os.path.join(p, f"{n}.yaml"), "w") as y_fp:
                file_struct = self.__get_yaml(p,n)
                y_fp.write(file_struct)
    
    def __get_yolo_box(sels, img_shape, box):
        iw, ih = img_shape[0], img_shape[1]
        x0, y0, w , h = box[0], box[1], box[2], box[3]
        return ((2*x0 + w)/(2*iw)), ((2*y0 + h)/(2*ih)), w/iw, h/ih

    def __map_category(self, dataset_name, class_idx):
        _classes = self.classes[dataset_name]
        if isinstance(_classes, (dict, list)):
            try:
                idx_pos = _classes.index(class_idx) if isinstance(_classes, list) else list(_classes.values()).index(int(class_idx))
            except ValueError:
                raise ValueError(f"Class index {class_idx} is not present in list")
        else:
            raise ValueError("Class structure type must be dict or list")
        
        return idx_pos
        
    def from_csv(self, train_csv, dataset_name,train_idx=None, test_csv=None):
        self.__prepare_directories()
        
        dataset_path = self.data[dataset_name]
        train_df = pd.read_csv(train_csv).dropna()
        unique_keys =  train_df["image_id"].unique()
        train_path = f"{dataset_path}/images/train"
        val_path = f"{dataset_path}/images/val"
        label_path = f"{dataset_path}/images/labels"
        
        if not train_idx:
            val_len = len(unique_keys) // 10
            train_idx = [i for i in range(len(unique_keys) - val_len)]
        else:
            if isinstance(train_idx, tuple):
                train_idx = [i for i in range(train_idx[0], train_idx[1])]
        print("Creating YOLO dataset....")
        for idx, img_id in enumerate(tqdm(unique_keys)):
            img_path = os.path.join(self.img_set, img_id + ".tif")
            img_shape = cv2.imread(img_path).shape[:2]
            targets = train_df[train_df['image_id'] == img_id]
            if idx in train_idx:
                ann_path = os.path.join(label_path, "train")
                shutil.copy(img_path, train_path)
            else:
                ann_path = os.path.join(label_path, "val")
                shutil.copy(img_path, val_path)
            
            with open(f"{ann_path}/{img_id}.txt", "w") as out_f:
                for box, cat in zip(targets["bbox"], targets["category_id"]):
                    try:
                        box = ast.literal_eval(box)
                    except:
                        raise ValueError(f"The current bounding box {box} should represent an array") 
                    center_x, center_y, w, h = self.__get_yolo_box(img_shape, box)
                    cat = self.__map_category(dataset_name, cat)
                    line = f"{cat} {center_x} {center_y} {w} {h}\n"
                    out_f.write(line)

        self.__prepare_yaml()
