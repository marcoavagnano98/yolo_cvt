import pandas as pd
import cv2
import os
import shutil
import ast

class YOLOConverter:
    def __init__(self, img_set, dataset_names, classes, train_idx = None):
        self.img_set = img_set
        self.indices = train_idx
        self.dataset_names = dataset_names
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

        for dataset_path in self.datasets_path:
            image_dirs = [os.path.join(dataset_path, "images"), os.path.join(dataset_path, "images", "train"),
                           os.path.join(dataset_path, "images", "val"), os.path.join(dataset_path, "labels")]
        ds = [parent_dir, self.node_dir] + self.datasets_path + image_dirs
        for d in ds:
            self.__make_dir(d)
        
    def __get_yaml(self, dataset_path,name):
        out = f"path: ../{dataset_path}\ntrain: images/train\nval: images/val\n\nnames:\n"
        _class  = self.classes[name]
        for idx, c in enumerate(_class):
            cline = f"{idx}: {c}"
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


    def from_csv(self, train_csv, dataset_name, test_csv=None):
        self.__prepare_directories()
        self.__prepare_yaml()
        
        train_df = pd.read_csv(train_csv).dropna()
        train_path = f"yolo/datasets/{dataset_name}/images/train"
        val_path = f"yolo/datasets/{dataset_name}/images/val"
        unique_keys =  train_df["image_id"].unique()
        
        if not self.indices:
            val_len = len(unique_keys) // 10
            train_idx = (0, len(unique_keys) - val_len)
        else:
            train_idx = self.indices

        dataset_path = self.data[dataset_name]
        for idx, img_id in enumerate(unique_keys):
            img_path = os.path.join(self.img_set, img_id + ".tif")
            img_shape = cv2.imread(img_path).shape[:2]
            targets = train_df[train_df['image_id'] == img_id]

            with open(f"{dataset_path}/labels/{img_id}.txt", "w") as out_f:
                for box, cat in zip(targets["bbox"], targets["category_id"]):
                    box = ast.literal_eval(box)
                    center_x, center_y, w, h = self.__get_yolo_box(img_shape, box)
                    line = f"{int(cat) - 1} {center_x} {center_y} {w} {h}\n"
                    out_f.write(line)

            if idx < train_idx[1] and idx >= train_idx[0]:
                shutil.copy(img_path, train_path)
            else:
                shutil.copy(img_path, val_path)


