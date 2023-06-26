from torch.utils.data import Dataset
from torch.utils.data import Sampler, BatchSampler
import os
from typing import List, Tuple, Dict
from PIL import Image
import torch, torch.nn as nn
import numpy as np
import pathlib
import h5py


def find_classes(directory: str, offset=0) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    class_to_idx = {cls_name: i + offset for i , cls_name in enumerate(classes)}
    return classes, class_to_idx



class CLIPBaseDataset(Dataset):
    
    def __init__(self, transform=None, clip_model=None, clip_preprocess=None, clip_device=None) -> None:
        
        super().__init__()
        self.transform = transform
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_device = clip_device
                    
    def generate_clip_image_feats(self, clip_model, start_idx) -> List[torch.Tensor]:
        idx = start_idx
        minib = 128
        next_1k = 1000
        result = []
        with torch.no_grad():
            while idx < self.len:
                next_idx = min(idx+minib, self.len)
                images = [self.load_image(i) for i in range(idx, next_idx)]
                if self.transform is not None:
                    images = [self.transform(img) for img in images]
                images = torch.stack(images)
                inputs_clip = self.clip_preprocess(images).to(self.clip_device)
                outputs_clip = self.clip_model.encode_image(inputs_clip).float().detach().cpu()
                for i in range(len(outputs_clip)):
                    result.append(outputs_clip[i])
                idx = next_idx
                if idx >= next_1k:
                    next_1k += 1000
                    print(f"Processed {idx} CLIP image features")   
        return result    
    
    def load_image(self, index: int) -> Image.Image:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        raise NotImplementedError()
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:    
        raise NotImplementedError() 
        
        

    
class CLIPImageBaseDataset(CLIPBaseDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.paths = []
        self.class_to_idx = {}
        self.clip_feats = []
        self.clip_caption_feats = []
        
    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path).convert('RGB') 
    
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, index: int):
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            img = self.transform(img)
            
        if self.clip_model is not None:
            if self.clip_caption_feats is not None:
                return (img, (self.clip_feats[index], self.clip_caption_feats[index])), class_idx
            else:
                return (img, self.clip_feats[index]), class_idx
        else:
            return img, class_idx # return data, label (X, y)  
        
        
        
     
    
        
class CLIPImageDataset(CLIPImageBaseDataset):
    
    def __init__(self, targ_dir: str, 
                 transform=None, clip_model=None, clip_preprocess=None, clip_device=None, use_caption=False) -> None:
        
        super().__init__(transform=transform, clip_model=clip_model, clip_preprocess=clip_preprocess, clip_device=clip_device)
        self.paths = sorted(list(pathlib.Path(targ_dir).glob("*/*.JPEG")))
        self.paths += sorted(list(pathlib.Path(targ_dir).glob("*/*.jpg")))
        self.len = len(self.paths)
        print("Dataset length:", self.len)
        self.classes, self.class_to_idx = find_classes(targ_dir)
        self.idx_to_class = {}
        for k, v in self.class_to_idx.items():
            self.idx_to_class[v] = k
        self.use_caption = use_caption
        
        image_feats_path = os.path.join(targ_dir, "clip_image_feats.pt")
        
        # Get CLIP image features
        self.clip_feats = None
        if self.clip_model is not None:
            if not os.path.exists(image_feats_path):
                clip_feats = self.generate_clip_image_feats(self.clip_model, 0)
                self.clip_feats = torch.stack(clip_feats)
                self.clip_feats_mean = self.clip_feats.mean(dim=0)
                torch.save(self.clip_feats, image_feats_path)
                print(f"Saved CLIP image features from {image_feats_path}")
            else:
                self.clip_feats = torch.load(image_feats_path)
                self.clip_feats_mean = self.clip_feats.mean(dim=0)
                print(f"Loaded CLIP image features from {image_feats_path}")
        
        # Get CLIP caption features if available
        if self.use_caption:
            caption_feats_path = os.path.join(targ_dir, "clip_caption_feats.pt")
            assert os.path.exists(caption_feats_path), "Please generate caption features first using the OFA repo and our ofa_gen_caption.py"
            self.clip_caption_feats = torch.load(caption_feats_path)
            print(f"Loaded CLIP caption features from {caption_feats_path}")  
        else:
            self.clip_caption_feats = None      
                        
        
        
class CLIPH5BaseDataset(CLIPBaseDataset):
    
    # Robotics dataset
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_rgbs = []
        self.hand_rgbs = []
        self.base_depths = []
        self.hand_depths = []
        self.labels = []
        self.clip_feats_base = []
        self.clip_feats_hand = []
        
    def generate_clip_image_feats(self, clip_model, start_idx, use_clip_region_feats) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Generate CLIP image features for base camera images and hand camera images in the robotics dataset
        idx = start_idx
        half_minib = 64
        next_1k = 1000
        result_base = []
        result_hand = []
        with torch.no_grad():
            while idx < self.len:
                next_idx = min(idx + half_minib, self.len)
                images = [self.base_rgbs[i] for i in range(idx, next_idx)]
                images.extend([self.hand_rgbs[i] for i in range(idx, next_idx)])
                images = torch.stack(images)
                inputs_clip = self.clip_preprocess(images).to(self.clip_device)
                if not use_clip_region_feats:
                    outputs_clip = self.clip_model.encode_image(inputs_clip).float().detach().cpu()
                else:
                    vs_model = self.clip_model.visual
                    x = vs_model.conv1(inputs_clip.type(self.clip_model.dtype))  # shape = [*, width, grid, grid]
                    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                    x = torch.cat([vs_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                    x = x + vs_model.positional_embedding.to(x.dtype)
                    x = vs_model.ln_pre(x)
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = vs_model.transformer(x)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = vs_model.ln_post(x[:, 1:, :])
                    if vs_model.proj is not None:
                        x = x @ vs_model.proj
                    grid_sz = int(np.sqrt(x.shape[1]))
                    x = x.reshape(x.shape[0], grid_sz, grid_sz, x.shape[-1]).permute(0,3,1,2)
                    if hasattr(self, 'avgpool'):
                        x = self.avgpool(x) # [*, width, 7, 7]
                    outputs_clip = x.float().detach().cpu()
                for i in range(next_idx - idx):
                    result_base.append(outputs_clip[i])
                    result_hand.append(outputs_clip[next_idx - idx + i])
                idx = next_idx
                if idx >= next_1k:
                    next_1k += 1000
                    print(f"Processed {idx} CLIP image features", flush=True)   
        return result_base, result_hand
    
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, index: int):
        base_rgb = self.base_rgbs[index]
        base_depth = self.base_depths[index]
        hand_rgb = self.hand_rgbs[index]
        hand_depth = self.hand_depths[index]
        im_return = (base_rgb, base_depth, hand_rgb, hand_depth)
        label = torch.zeros(self.num_classes, dtype=torch.long)
        label[self.labels[index]] = 1
            
        if self.clip_model is not None:
            return (im_return, (self.clip_feats_base[index], self.clip_feats_hand[index])), label
        else:
            return im_return, label    
                
                
                
                
class CLIPH5Dataset(CLIPH5BaseDataset):
    
    # Robotics dataset
    def __init__(self, h5_path: str, raw_classes: List[str], 
                 transform=None, clip_model=None, clip_preprocess=None, clip_device=None,
                 mode=None, use_clip_region_feats=False) -> None:
        
        super().__init__(transform=transform, clip_model=clip_model, clip_preprocess=clip_preprocess, clip_device=clip_device)
        assert self.transform is not None
        self.raw_classes = raw_classes
        self.cls_preprocess_fxn = lambda x: x[x.find('_')+1:].replace('_', ' ') # extract object names
        self.classes = [self.cls_preprocess_fxn(x) for x in raw_classes]
        
        self.idx_to_class = {}
        self.class_to_idx = {}
        for i, clas in enumerate(self.classes):
            self.idx_to_class[i] = clas
            self.class_to_idx[clas] = i
        self.num_classes = len(self.classes)
            
        self.h5 = h5py.File(h5_path, 'r')
        self.base_rgbs = []
        self.base_depths = []
        self.hand_rgbs = []
        self.hand_depths = []
        self.labels = []
        
        next_1000 = 1000
        idx = 0
        print(len(self.h5.keys()))
        for k in sorted(self.h5.keys()):
            idx += 1
            if idx >= next_1000:
                print(f"Loaded {next_1000} images")
                next_1000 += 1000
            traj = self.h5[k]
            if mode is not None and hasattr(traj, 'traj_type') and traj['traj_type'].decode('utf-8') != mode:
                continue
            base_rgb = np.uint8(traj['dict_str_base_camera']['dict_str_rgb']) # [H, W, 3]
            base_depth = np.float32(traj['dict_str_base_camera']['dict_str_depth']) # [H, W, 1]
            hand_rgb = np.uint8(traj['dict_str_hand_camera']['dict_str_rgb'])
            hand_depth = np.float32(traj['dict_str_hand_camera']['dict_str_depth'])
            
            self.base_rgbs.append(self.transform(base_rgb))
            self.base_depths.append(self.transform(base_depth))
            self.hand_rgbs.append(self.transform(hand_rgb))
            self.hand_depths.append(self.transform(hand_depth))
            self.labels.append(torch.LongTensor([self.class_to_idx[self.cls_preprocess_fxn(x.decode('utf-8'))] for x in traj['model_ids']]))
            
        self.len = len(self.base_rgbs)
        print("Dataset length:", self.len)
        
        # Get CLIP image features
        self.use_clip_region_feats = use_clip_region_feats
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        image_feats_path = os.path.join(
            os.path.dirname(h5_path), 
            "clip_image_feats.pt" if not use_clip_region_feats else "clip_image_region_feats.pt"
        )
        self.clip_feats_base = self.clip_feats_hand = None
        if self.clip_model is not None:
            if not os.path.exists(image_feats_path):
                clip_feats_base, clip_feats_hand = self.generate_clip_image_feats(self.clip_model, 0, use_clip_region_feats)
                self.clip_feats_base = torch.stack(clip_feats_base)
                self.clip_feats_hand = torch.stack(clip_feats_hand)
                torch.save(torch.stack([self.clip_feats_base, self.clip_feats_hand]), image_feats_path)
                print(f"Saved CLIP image features to {image_feats_path}")
            else:
                self.clip_feats_base, self.clip_feats_hand = torch.load(image_feats_path)
                print(f"Loaded CLIP image features from {image_feats_path}")
                        
        
        
        
        
        
        
class FSCLIPImageDataset(CLIPImageBaseDataset):
    
    def __init__(self, targ_dir: str, fs_dir: str, fs_num: int=0, fs_temp_dir: str='/tmp/val',
                 transform=None, clip_model=None, clip_preprocess=None, clip_device=None, use_caption=False) -> None:
        
        super().__init__(transform=transform, clip_model=clip_model, clip_preprocess=clip_preprocess, clip_device=clip_device)
        self.paths = sorted(list(pathlib.Path(targ_dir).glob("*/*.JPEG"))) # all training paths
        self.paths += sorted(list(pathlib.Path(targ_dir).glob("*/*.jpg")))
        self.fs_num = 0
        self.fs_paths = []
        
        self.valdir = fs_temp_dir # temporary directory to store validation images after removing few-shot images
        if os.path.exists(self.valdir):
            import shutil
            shutil.rmtree(self.valdir)
            
        # Generate few-shot examples
        for category in sorted(os.listdir(fs_dir)):
            fs_category_dir = os.path.join(fs_dir, category)
            if not os.path.isdir(fs_category_dir):
                continue
            val_category_dir = os.path.join(self.valdir, category)
            if not os.path.exists(val_category_dir):
                os.makedirs(val_category_dir)
            category_val_paths = sorted(list(pathlib.Path(category_dir).glob("*.JPEG")))
            category_val_paths += sorted(list(pathlib.Path(category_dir).glob("*.jpg")))
            fs_train_paths = category_val_paths[:fs_num]
            category_val_paths = category_val_paths[fs_num:]
            self.fs_paths += fs_train_paths
            self.fs_num += fs_num
            for val_path in category_val_paths:
                os.symlink(val_path.resolve(), os.path.join(val_category_dir, val_path.name))
        self.fs_paths = sorted(self.fs_paths)
        self.paths += self.fs_paths
        
        self.len = len(self.paths)
        print("Dataset length:", self.len)
        classes, class_to_idx = find_classes(targ_dir)
        fs_classes, fs_class_to_idx = find_classes(fs_dir, offset=len(classes))
        self.classes = classes + fs_classes
        class_to_idx.update(fs_class_to_idx)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {}
        for k, v in self.class_to_idx.items():
            self.idx_to_class[v] = k
        
        # Get CLIP image features
        image_feats_path = os.path.join(targ_dir, "clip_image_feats.pt")
        self.clip_feats = None
        if self.clip_model is not None:
            if not os.path.exists(image_feats_path):
                clip_feats = self.generate_clip_image_feats(self.clip_model, 0)
                self.clip_feats = torch.stack(clip_feats)
                torch.save(self.clip_feats, image_feats_path)
                print(f"Saved CLIP image features to {image_feats_path}")
            else:
                self.clip_feats = torch.load(image_feats_path)
                print(f"Loaded CLIP image features from {image_feats_path}")
                if len(self.clip_feats) < self.len:
                    print("==== Warning: CLIP image features are incomplete. Generating missing features ====")
                    clip_feats = self.generate_clip_image_feats(self.clip_model, len(self.clip_feats))
                    self.clip_feats = torch.cat([self.clip_feats, torch.stack(clip_feats)], dim=0)
        
        # Get CLIP caption features if available
        self.use_caption = use_caption
        if self.use_caption:
            caption_feats_path = os.path.join(targ_dir, "clip_caption_feats.pt")
            assert os.path.exists(caption_feats_path), "Please generate caption features first using the OFA repo and our ofa_gen_caption.py"
            self.clip_caption_feats = torch.load(caption_feats_path)
            fs_caption_feats_path = os.path.join(fs_dir, f"clip_caption_feats_{fs_num}_shot.pt")
            assert os.path.exists(fs_caption_feats_path), "Few-shot caption features don't exist"
            fs_caption_feats = torch.load(fs_caption_feats_path)
            assert fs_caption_feats.shape[0] == len(self.fs_paths), f"{fs_caption_feats.shape[0]} != {len(self.fs_paths)}"
            self.clip_caption_feats = torch.cat([self.clip_caption_feats, fs_caption_feats], dim=0)
            print("Loaded CLIP caption features from disk")  
        else:
            self.clip_caption_feats = None          
        






class FSCLIPH5Dataset(CLIPH5BaseDataset):
    
    def __init__(self, train_h5_path: str, val_h5_path: str, raw_classes: List[str], fs_num: int=0,
                 transform=None, clip_model=None, clip_preprocess=None, clip_device=None,
                 use_clip_region_feats=False) -> None:
        
        super().__init__(transform=transform, clip_model=clip_model, clip_preprocess=clip_preprocess, clip_device=clip_device)
        assert self.transform is not None
        self.raw_classes = raw_classes
        self.cls_preprocess_fxn = lambda x: x[x.find('_')+1:].replace('_', ' ') # extract object names
        self.classes = [self.cls_preprocess_fxn(x) for x in raw_classes]
        
        self.idx_to_class = {}
        self.class_to_idx = {}
        for i, clas in enumerate(self.classes):
            self.idx_to_class[i] = clas
            self.class_to_idx[clas] = i
        self.num_classes = len(self.classes)
            
        self.train_h5 = h5py.File(train_h5_path, 'r')
        self.val_h5 = h5py.File(val_h5_path, 'r')
        self.base_rgbs = []
        self.base_depths = []
        self.hand_rgbs = []
        self.hand_depths = []
        self.labels = []
        next_1000 = 1000
        idx = -1
        
        keys_to_use = list(sorted(self.train_h5.keys())) + list(sorted(self.val_h5.keys()))[:fs_num]
        self.len = len(keys_to_use)
        self.fs_num = fs_num
        print("Dataset length:", self.len, "Containing Few Shot Samples:", self.fs_num)
        for k in keys_to_use:
            idx += 1
            if idx >= next_1000:
                print(f"Loaded {next_1000} images", flush=True)
                next_1000 += 1000
            if idx < self.len - self.fs_num:
                traj = self.train_h5[k]
            else:
                traj = self.val_h5[k]
            base_rgb = np.uint8(traj['dict_str_base_camera']['dict_str_rgb']) # [H, W, 3]
            base_depth = np.float32(traj['dict_str_base_camera']['dict_str_depth']) # [H, W, 1]
            hand_rgb = np.uint8(traj['dict_str_hand_camera']['dict_str_rgb'])
            hand_depth = np.float32(traj['dict_str_hand_camera']['dict_str_depth'])
            
            self.base_rgbs.append(self.transform(base_rgb))
            self.base_depths.append(self.transform(base_depth))
            self.hand_rgbs.append(self.transform(hand_rgb))
            self.hand_depths.append(self.transform(hand_depth))
            self.labels.append(torch.LongTensor([self.class_to_idx[self.cls_preprocess_fxn(x.decode('utf-8'))] for x in traj['model_ids']]))
        
        # Get CLIP image features
        image_feats_path = os.path.join(
            os.path.dirname(train_h5_path), 
            "clip_image_feats.pt" if not use_clip_region_feats else "clip_image_region_feats.pt"
        )
        self.use_clip_region_feats = use_clip_region_feats
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.clip_feats_base = self.clip_feats_hand = None
        if self.clip_model is not None:
            if not os.path.exists(image_feats_path):
                clip_feats_base, clip_feats_hand = self.generate_clip_image_feats(self.clip_model, 0, use_clip_region_feats)
                self.clip_feats_base = torch.stack(clip_feats_base)
                self.clip_feats_hand = torch.stack(clip_feats_hand)
                torch.save(torch.stack([self.clip_feats_base, self.clip_feats_hand]), image_feats_path)
                print(f"Saved CLIP image features to {image_feats_path}")
            else:
                self.clip_feats_base, self.clip_feats_hand = torch.load(image_feats_path)
                print(f"Loaded CLIP image features from {image_feats_path}")
                if len(self.clip_feats_base) < self.len:
                    print("==== Warning: CLIP image features are incomplete. Generating missing features ====")
                    clip_feats_base, clip_feats_hand = self.generate_clip_image_feats(self.clip_model, len(self.clip_feats_base), use_clip_region_feats)
                    self.clip_feats_base = torch.cat([self.clip_feats_base, torch.stack(clip_feats_base)], dim=0)
                    self.clip_feats_hand = torch.cat([self.clip_feats_hand, torch.stack(clip_feats_hand)], dim=0)
                assert len(self.clip_feats_base) == self.len




class FewShotSampler(Sampler):
    def __init__(self, dataset, batch_size, samples_per_gpu=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_gpu = samples_per_gpu
        if dataset.fs_num > batch_size // 2:
            self.num_samples = (len(dataset) - dataset.fs_num) // (batch_size // 2) * batch_size + \
                               (len(dataset) - dataset.fs_num) % (batch_size // 2) + batch_size // 2
        else:
            self.num_samples = batch_size * ((len(dataset) - dataset.fs_num) // (batch_size - dataset.fs_num)) + \
                               (len(dataset) - dataset.fs_num) % (batch_size - dataset.fs_num) + dataset.fs_num
        self.num_samples = int(np.ceil(self.num_samples / self.samples_per_gpu) * self.samples_per_gpu)

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        fs_indices = indices[-self.dataset.fs_num:]
        indices = indices[:-self.dataset.fs_num]
        np.random.shuffle(indices)
        result_indices = []
        half_batch_size = self.batch_size // 2
        if self.dataset.fs_num > half_batch_size:
            for batch_id in range(len(indices) // half_batch_size + 1):
                batch_indices = np.random.choice(fs_indices, half_batch_size, replace=False)
                batch_indices = np.concatenate([batch_indices, indices[batch_id * half_batch_size:(batch_id + 1) * half_batch_size]], axis=0)
                np.random.shuffle(batch_indices)
                result_indices.append(batch_indices)
        else:
            num_extra = self.batch_size - self.dataset.fs_num
            for batch_id in range(len(indices) // num_extra + 1):
                batch_indices = indices[batch_id * num_extra: (batch_id + 1) * num_extra]
                batch_indices = np.concatenate([fs_indices, batch_indices], axis=0)
                np.random.shuffle(batch_indices)
                result_indices.append(batch_indices)
        indices = result_indices
        # num_extra = self.num_samples - len(self.dataset)
        # indices = np.concatenate([indices, np.random.choice(indices, num_extra)])
        indices = np.concatenate(indices, axis=-1)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)
    
    def __len__(self):
        return self.num_samples