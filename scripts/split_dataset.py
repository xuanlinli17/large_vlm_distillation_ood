import os
import shutil
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data-root', type=str, required=True)
parser.add_argument('--dataset-name', type=str, required=True, choices=["CaltechBirds", "Flower102", "Food101", "StanfordCars", "SUN397", "tiered-Image Net"])

args = parser.parse_args()

dataset_prefix = {"CaltechBirds": "CUB_200_2011/images",
                  "Flower102": "jpg",
                  "Food101": "images",
                  "StanfordCars": "cars_train",
                  "SUN397": "",
                  "tiered-ImageNet": "tiered_imagenet/train"}

dataset_root = Path(args.data_root)
if not dataset_root.exists():
    raise ValueError(f"Dataset {args.dataset_name} not found in {args.data_root}")
print(f"Processing {args.dataset_name} from {args.data_root}")
for split in ["train", "val", "val_on_train"]:
    split_root = dataset_root / split
    if not split_root.exists():
        os.mkdir(split_root)
    img_list = []
    with open(f"data/{args.dataset_name}/{split}_data.txt", 'r') as f: # load image split info from our repo
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            img_list.append(line)
    for img_path in img_list:
        img_path = Path(img_path)
        class_name = img_path.parent.name
        class_dir = split_root / class_name
        if not class_dir.exists():
            os.mkdir(class_dir)
        if args.dataset_name == "SUN397":
            prefix = class_name[0]
            class_name_map = {}
            with open(dataset_root/"ClassName.txt", 'r') as f:
                for line in f:
                    line = line[3:].strip()
                    class_name_map[line.replace('/', '_')] = line
            shutil.copyfile(dataset_root / dataset_prefix[args.dataset_name] / prefix / class_name_map[class_name] / img_path.name, class_dir / img_path.name)
        else:
            try:
                shutil.copyfile(dataset_root / dataset_prefix[args.dataset_name] / class_name / img_path.name, class_dir / img_path.name)
            except:
                shutil.copyfile(dataset_root / dataset_prefix[args.dataset_name] / img_path.name, class_dir / img_path.name)