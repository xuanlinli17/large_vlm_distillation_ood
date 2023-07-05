import os
import shutil
from mmengine import load, dump
from pathlib import Path

data_root = "./data"
default_datasets = ["CaltechBirds", "Flower102", "Food101", "StanfordCars", "SUN397", "tiered-Image Net"]
dataset_prefix = {"CaltechBirds", "CUB_200_2011/images",
                  "Flower102", "jpg",
                  "Food101", "",
                  "StanfordCars", "",
                  "SUN397", "",
                  "tiered-ImageNet", ""}

for dataset in default_datasets:
    dataset_root = Path(data_root) / dataset
    if not data_root.exists():
        raise ValueError(f"Dataset {dataset} not found in {data_root}")
    print(f"Processing {dataset}")
    for split in ["train", "val", "val_on_train"]:
        split_root = dataset_root / split
        if not split_root.exists():
            os.mkdir(split_root)
        img_list = load(data_root / f"{split}_data.txt")
        for img_path in img_list:
            img_path = Path(img_path)
            class_name = img_path.parent.name
            class_dir = split_root / class_name
            if not class_dir.exists():
                os.mkdir(class_dir)
            try:
                shutil.copyfile(data_root / dataset_prefix[dataset] / class_name / img_path.name, class_dir / img_path.name)
            except:
                raise ValueError(f"Image {img_path} not found in {data_root / dataset_prefix[dataset]}")