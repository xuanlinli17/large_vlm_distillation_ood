from pathlib import Path
parent = Path("/datasets/tiered_imagenet/splited/")
for split in ["train", "val", "val_on_train"]:
    split_data = []
    split_root = parent / split
    for class_dir in split_root.iterdir():
        if ".pt" in class_dir.name:
            continue
        for image_path in class_dir.iterdir():
            split_data.append(f"{split}/{class_dir.name}/{image_path.name}")
    with open(f"{split}_data.txt", "w") as f:
        f.write("\n".join(split_data))