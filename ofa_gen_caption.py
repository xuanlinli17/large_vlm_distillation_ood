
from __future__ import print_function
"""
pip install pip==21.2.4  
pip install setuptools==59.5.0
pip install fairseq --no-deps
pip install tensorboard
pip install timm
pip install einops
pip uninstall torch torchvision -y
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/openai/CLIP.git
!wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/caption_base_best.pt
!mv caption_base_best.pt checkpoints/caption_base_best.pt
"""
import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import pathlib

# Parse arguments
parser = argparse.ArgumentParser(description='CLIP forward imagenet')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('--few-shot-num', type=int, default=-1, help="few shot setting")
# Openset-specific
parser.add_argument('--clip-repo', type=str, default='clip', choices=['clip', 'open_clip'])
parser.add_argument('--clip-model', type=str, default='ViT-L/14')
parser.add_argument('--clip-dataset', type=str, default='openai', choices=['openai', 'laion400m_e31', 'laion400m_e32', 'laion2b_s32b_b82k'])
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()



    

def main():
    clip_device = f'cuda:{args.gpu_id}'
    print("clip_device", clip_device)
    if args.clip_repo == 'clip':
        import clip
        clip_model, clip_preprocess = clip.load(args.clip_model, device=clip_device)
        if 'ViT' in args.clip_model or args.clip_model in ['RN50']:
            clip_preprocess = transforms.Compose([
                transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
                transforms.CenterCrop(size=(224, 224)),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])           
        elif args.clip_model == 'RN50x16':
            clip_preprocess = transforms.Compose([
                transforms.Resize(size=384, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
                transforms.CenterCrop(size=(384, 384)),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            raise NotImplementedError()
    elif args.clip_repo == 'open_clip':
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_dataset)
        clip_preprocess = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.CenterCrop(size=(224, 224)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])       
    print("clip_preprocess", clip_preprocess)
    clip_model.to(clip_device).eval()
    for m in clip_model.parameters():
        m.requires_grad = False
        
    if args.few_shot_num > 0:
        images = []
        for category in sorted(os.listdir(args.data)):
            category_dir = os.path.join(args.data, category)
            fs_train_paths = sorted(list(pathlib.Path(category_dir).glob("*.JPEG")))[:args.few_shot_num]
            fs_train_paths += sorted(list(pathlib.Path(category_dir).glob("*.jpg")))[:args.few_shot_num]
            images += fs_train_paths
        print("few shot images:", images)
    else:
        images = sorted(list(pathlib.Path(args.data).glob("*/*.JPEG"))) # note: you'd have to update this if you've got .png's or .jpeg's
        images += sorted(list(pathlib.Path(args.data).glob("*/*.jpg")))
            
    clip_caption_feats = None
    idx = 0
    minib = 32
    next_1k = 1000
                
    from fairseq import utils, tasks
    from fairseq import checkpoint_utils
    from utils.eval_utils import eval_step
    from tasks.mm_tasks.caption import CaptionTask
    from models.ofa import OFAModel
    # Register refcoco task
    tasks.register_task('caption', CaptionTask)
    use_fp16 = True
    overrides={"eval_cider":False, "beam":5, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                        utils.split_paths('checkpoints/caption_base_best.pt'),
                        # utils.split_paths('checkpoints/caption_ofa_medium.pt'),
                        arg_overrides=overrides
    )
    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)
    # Initialize generator
    generator = task.build_generator(models, cfg.generation)                
                
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose(
        [
            lambda image: image.convert("RGB"),
            transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()
    def encode_text(text, length=70, append_bos=False, append_eos=False):
        s = task.tgt_dict.encode_line(
            line=task.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([bos_item, s])
        if append_eos:
            s = torch.cat([s, eos_item])
        return s

    # Construct input for caption task
    def construct_samples(ids, images):
        
        batch_size = len(images)
        patch_images = torch.stack([patch_resize_transform(Image.open(image)) for image in images])
        patch_masks = torch.tensor([True]).expand(batch_size, )
        src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
        # src_text = encode_text(" describe the Shih-Tzu in this image: ", append_bos=True, append_eos=True).unsqueeze(0)
        src_text = src_text.expand(batch_size, -1)
        src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
        samples = {
            "id":np.array(ids),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_images,
                "patch_masks": patch_masks
            }
        }
        return samples
    
    # Function to turn FP32 to FP16
    def apply_half(t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t       
    
             
    from utils.eval_utils import eval_caption
    clip_caption_feats = []
    next_1k = 1000
    batch_size = 32
    while idx < len(images):
        if idx > next_1k:
            print(idx)
            next_1k += 1000
        cur_batch_end = min(len(images), idx + batch_size)
        print(cur_batch_end)
        sample = construct_samples(np.arange(idx, cur_batch_end), images[idx:cur_batch_end])
        # (32,) torch.Size([32, 17]) torch.Size([32]) torch.Size([32, 3, 384, 384]) torch.Size([32])
        # print(id_cur_batch.shape, src_text_cur_batch.size(), src_length_cur_batch.size(), patch_image_cur_batch.size(), patch_mask_cur_batch.size())
        sample = utils.move_to_cuda(sample)
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
        with torch.no_grad():
            try:
                result, scores = eval_caption(task, generator, models, sample)
                clip_caption_feats.extend([x['caption'] for x in result])
            except:
                for i in range(idx, cur_batch_end):
                    sample = construct_samples(np.arange(i, i+1), images[i:i+1])
                    sample = utils.move_to_cuda(sample)
                    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
                    try:
                        result, scores = eval_caption(task, generator, models, sample)
                        clip_caption_feats.extend([x['caption'] for x in result])
                    except:
                        clip_caption_feats.append('a photo')
        idx = cur_batch_end
        
    batch_size = 128
    idx = 0
    final = []
    while idx < len(images):
        cur_batch_end = min(len(images), idx + batch_size)
        if args.clip_repo == 'clip': 
            cur_text_features = clip_model.encode_text(clip.tokenize(clip_caption_feats[idx:cur_batch_end]).to(clip_device)).float().detach()
        elif args.clip_repo == 'open_clip':
            tokenize = open_clip.tokenizer.tokenize
            cur_text_features = clip_model.encode_text(tokenize(clip_caption_feats[idx:cur_batch_end]).to(clip_device)).float().detach()
        final.append(cur_text_features)
        idx = cur_batch_end
    final = torch.cat(final, dim=0).detach().cpu()
    
    if args.few_shot_num > 0:
        save_path = f"clip_caption_feats_{args.few_shot_num}_shot.pt"
    else:
        save_path = "clip_caption_feats.pt"
    torch.save(final, os.path.join(args.data, save_path))
                
                
if __name__ == '__main__':
    main()