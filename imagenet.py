'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017

python imagenet.py -a resnet18 --data /home/xuanlin/ILSVRC/Data/split_6_4/ --epochs 90 --lr 0.1 --schedule 30 60 --gamma 0.1 -c checkpoints/imagenet/split_6_4/resnet18 \
    --use-clip --clip-align-image-classification=1 --label-path=/home/xuanlin/ILSVRC/label2text.txt --clip-gpu-id=0 \
    --resume checkpoints/imagenet/split_6_4/resnet18/model_best.pth.tar --evaluate
    
python imagenet.py -a resnet18 --data /home/xuanlin/ILSVRC/Data/split_6_4/ --epochs 100 --lr 0.00001 --schedule 30 60 --gamma 0.1 -c checkpoints/imagenet/split_6_4/resnet18_clipalignimage_contrastive_no_mse/tmp \
    --use-clip --label-path=/home/xuanlin/ILSVRC/label2text.txt --clip-gpu-id=0 \
    --clip-align-image-classification=0 --clip-align-image-contrastive --clip-align-image-mse \
    --resume checkpoints/imagenet/split_6_4/resnet18_clipalignimage_contrastive_no_mse/checkpoint.pth.tar
    
python imagenet.py -a resnet18 --data /home/xuanlin/ILSVRC/Data/split_6_4/ --epochs 90 --lr 0.1 --schedule 30 60 --gamma 0.1 -c checkpoints/imagenet/split_6_4/resnet18_clipalignimage \
    --use-clip --label-path=/home/xuanlin/ILSVRC/label2text.txt --clip-gpu-id=0 \
    --clip-align-image-classification=0 --clip-align-image-mse
    
python imagenet.py -a resnet18 --data /home/xuanlin/ILSVRC/Data/split_6_4/ --epochs 90 --lr 0.1 --schedule 30 60 --gamma 0.1 -c checkpoints/imagenet/split_6_4/resnet18_clipalignimage_mse_unnorm \
    --use-clip --label-path=/home/xuanlin/ILSVRC/label2text.txt --clip-gpu-id=0 \
    --clip-align-image-classification=0 --clip-align-image-mse-unnorm
    
python imagenet.py -a resnet18 --data /home/xuanlin/ILSVRC/Data/split_6_4/ --epochs 90 --lr 0.1 --schedule 30 60 --gamma 0.1 -c checkpoints/imagenet/split_6_4/resnet18_clipalignimage_classification \
    --use-clip --label-path=/home/xuanlin/ILSVRC/label2text.txt --clip-gpu-id=0 \
    --clip-align-image-classification=1 --clip-align-image-mse
    
python imagenet.py -a resnet18 --data /home/xuanlin/ILSVRC/Data/split_6_4/ --epochs 90 --lr 0.1 --schedule 30 60 --gamma 0.1 -c checkpoints/imagenet/split_6_4/resnet18_clipalignimage_contrastive \
    --use-clip --label-path=/home/xuanlin/ILSVRC/label2text.txt --clip-gpu-id=0 \
    --clip-align-image-classification=0 --clip-align-image-contrastive --clip-align-image-mse
    
python imagenet.py -a resnet18 --data /home/xuanlin/ILSVRC/Data/split_6_4/ --epochs 90 --lr 0.1 --schedule 30 60 --gamma 0.1 -c checkpoints/imagenet/split_6_4/resnet18_clipalignimage_contrastive_no_mse \
    --use-clip --label-path=/home/xuanlin/ILSVRC/label2text.txt --clip-gpu-id=0 \
    --clip-align-image-classification=0 --clip-align-image-contrastive
    
python imagenet.py -a resnet18 --data /home/xuanlin/ILSVRC/Data/split_6_4/ --epochs 90 --lr 0.1 --schedule 30 60 --gamma 0.1 -c checkpoints/imagenet/split_6_4/resnet18_clipalignimage_contrastive_only_other_no_mse \
    --use-clip --label-path=/home/xuanlin/ILSVRC/label2text.txt --clip-gpu-id=0 \
    --clip-align-image-classification=0 --clip-align-image-contrastive-only-other
    
python imagenet.py -a resnet18 --data /home/xuanlin/ILSVRC/Data/split_6_4/ --epochs 90 --lr 0.001 --onecyclelr -c checkpoints/imagenet/split_6_4/resnet18 \
    --use-clip --clip-align-image-classification=1 --label-path=/home/xuanlin/ILSVRC/label2text.txt --clip-gpu-id=0
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models

from custom_data_loader import CLIPImageDataset

import numpy as np

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--repeat-epochs', default=1, type=int, metavar='N',
                    help='repeat training batch in the same epoch')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--skip-val', action='store_true', help='skip validation during training')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--onecyclelr', action='store_true', help='use onecyclelr')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--log-avg', type=int, default=1, help='log average of last 5 batches')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Openset-specific
parser.add_argument('--use-clip', action='store_true', help='whether to use CLIP model')
parser.add_argument('--clip-repo', type=str, default='clip', choices=['clip', 'open_clip'])
parser.add_argument('--clip-model', type=str, default='ViT-L/14')
parser.add_argument('--clip-dataset', type=str, default='openai', choices=['openai', 'laion400m_e31', 'laion400m_e32', 'laion2b_s32b_b82k'])
parser.add_argument('--clip-align-image-classification', type=int, default=1)
parser.add_argument('--chatgpt-raw-text-file', type=str, default=None)
parser.add_argument('--rand-drop-text', action='store_true')
parser.add_argument('--chatgpt-text-features', type=str, default=None)
parser.add_argument('--chatgpt-ensemble', action='store_true')
parser.add_argument('--clip-align-proximal-image-num', type=int, default=-1)
parser.add_argument('--clip-align-proximal-text-num', type=int, default=-1)
parser.add_argument('--clip-align-proximal-text-axis', type=int, default=1, choices=[0, 1])
parser.add_argument('--clip-align-text-only-proximal', action='store_true')
parser.add_argument('--clip-filter-out-wrong-alignment', action='store_true')
parser.add_argument('--clip-align-image-aux-caption', action='store_true')
parser.add_argument('--clip-align-image-mse', action='store_true')
parser.add_argument('--clip-align-image-mse-unnorm', action='store_true')
parser.add_argument('--clip-align-image-global-bias', action='store_true')
parser.add_argument('--clip-align-image-global-bias-custom', action='store_true')
parser.add_argument('--clip-align-image-contrastive', action='store_true')
parser.add_argument('--clip-align-image-contrastive-mode', type=str, default='bidirectional', choices=['single', 'bidirectional', 'kl'])
parser.add_argument('--clip-align-image-contrastive-random-combine', action='store_true')
parser.add_argument('--clip-align-image-contrastive-only-other', action='store_true')
parser.add_argument('--clip-align-image-contrastive-hard-sample', action='store_true')
parser.add_argument('--clip-align-image-contrastive-prototype', action='store_true')
parser.add_argument('--clip-align-image-contrastive-projection', action='store_true')
parser.add_argument('--clip-align-image-relative-vector', type=str, default=None)
parser.add_argument('--clip-align-text-hinge', action='store_true')
parser.add_argument('--label-path', type=str, default=None, help='path to label text file')
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--clip-align-image-contrastive-adaptive-temperature', action='store_true', help='whether to use adaptive temperature')
parser.add_argument('--few-shot-num', type=int, default=0, help='number of few-shot examples')
parser.add_argument('--few-shot-method', type=str, default='None', help='few-shot mode, support retrieval or finetune')
parser.add_argument('--ood-text', type=str, default=None, help='path to OOD text file')
# Miscs
parser.add_argument('--closed-set-contrastive', action='store_true', help='whether to use closed set contrastive learning')
parser.add_argument('--prompt-learner', action='store_true', help='whether to use prompt learner (CoOp)')
parser.add_argument('--prompt-learner-nctx', type=int, default=8, help='number of CoOp context tokens')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--use-adam', action='store_true', help='whether to use adam optimizer')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--clip-gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES for the CLIP model')

args = parser.parse_args()
args.clip_align_image_classification = bool(args.clip_align_image_classification)
print("clip_align_image_classification = ", args.clip_align_image_classification)
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
assert use_cuda

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    cuda_device = f"cuda:{args.gpu_id}"
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.use_clip:
        clip_device = f'cuda:{args.clip_gpu_id}'
        print("clip_device", clip_device)
        if args.clip_repo == 'clip':
            import clip
            clip_model, clip_preprocess_orig = clip.load(args.clip_model, device=clip_device)
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
            clip_model, _, clip_preprocess_orig = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_dataset)
            clip_preprocess = transforms.Compose([
                transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
                transforms.CenterCrop(size=(224, 224)),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])       
        else:
            raise NotImplementedError(f"Unknown CLIP repo: {args.clip_repo}")
        print("clip_preprocess", clip_preprocess)
        clip_model.to(clip_device).eval()
        clip_model = clip_model.to(torch.float32)
        # DO NOT UPDATE CLIP MODEL
        for m in clip_model.parameters():
            m.requires_grad = False
    else:
        clip_model = clip_preprocess = clip_device = None
        
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    val_on_train_dir = os.path.join(args.data, 'val_on_train')
    
    if not os.path.exists(val_on_train_dir):
        val_on_train_dir = None
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    if 'vit' not in args.arch:
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                normalize,
            ])
        pre_train_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
                transforms.Lambda(lambda img: (img * 255.0).to(torch.uint8)),
                transforms.RandAugment(2, 9),
                transforms.Lambda(lambda img: (img / 255.0).to(torch.float32)),
                transforms.RandomResizedCrop(224),
                normalize,
            ])
        pre_train_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor()
        ])
        
    if not args.evaluate:
        if args.few_shot_num > 0 and args.few_shot_method == 'finetune':
            from custom_data_loader import FSCLIPImageDataset
            train_dataset = FSCLIPImageDataset(traindir,
                                               fs_dir=valdir,
                                               fs_num=args.few_shot_num,
                                               transform=pre_train_transform,
                                               clip_model=clip_model,
                                               clip_preprocess=clip_preprocess,
                                               clip_device=clip_device,
                                               use_caption=args.clip_align_image_aux_caption)
            valdir = train_dataset.valdir
        else:
            train_dataset = CLIPImageDataset(traindir, 
                                            pre_train_transform,
                                            clip_model=clip_model,
                                            clip_preprocess=clip_preprocess,
                                            clip_device=clip_device,
                                            use_caption=args.clip_align_image_aux_caption,)
    else:
        # train_dataset = datasets.ImageFolder(traindir, pre_train_transform)
        train_dataset = CLIPImageDataset(traindir, 
                                        pre_train_transform,
                                        clip_model=None,
                                        clip_preprocess=None,
                                        clip_device=None,
                                        use_caption=False)
    
    if args.few_shot_num > 0 and args.few_shot_method == 'finetune':
        from custom_data_loader import FewShotSampler
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch,
            num_workers=args.workers, pin_memory=True,
            sampler=FewShotSampler(train_dataset, args.train_batch))
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    val_transform = transforms.Compose([
            transforms.CenterCrop(224),
            normalize,
        ])
    val_dataset = CLIPImageDataset(valdir, 
                                   transforms.Compose([
                                        transforms.Resize([256, 256]),
                                        transforms.ToTensor()
                                    ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    val_on_train_dataset = None
    val_on_train_loader = None
    if val_on_train_dir is not None:
        val_on_train_dataset = CLIPImageDataset(val_on_train_dir, 
                                                transforms.Compose([
                                                    transforms.Resize([256, 256]),
                                                    transforms.ToTensor()
                                                ]))
        val_on_train_loader = torch.utils.data.DataLoader(
            val_on_train_dataset, 
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # create model
    extra_args = dict()
    if args.clip_model == 'ViT-B/32':
        clip_feats_dim = 512
    elif args.clip_model in ['RN50']:
        clip_feats_dim = 1024
    else:
        clip_feats_dim = 768
    if args.use_clip:
        extra_args['fc_out_dim'] = clip_feats_dim # match clip feature dimension
    if args.clip_align_image_contrastive_projection:
        extra_args['fc_out_dim'] = clip_feats_dim
    if ('efficientnet' in args.arch or 'vit' in args.arch) and 'fc_out_dim' in extra_args:
        extra_args['num_classes'] = extra_args['fc_out_dim'] # "num_classes" here refers to feature dim; we are not doing regular cross entropy here
        extra_args.pop('fc_out_dim')
        
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if 'efficientnet' in args.arch:
            extra_args['advprop'] = args.advprop
            # model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop, **extra_args)
        model = models.__dict__[args.arch](pretrained=True, **extra_args)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                    **extra_args
                )
    # elif 'efficientnet' in args.arch:
    #     model = EfficientNet.from_name(args.arch, **extra_args)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](**extra_args)
        
    if args.closed_set_contrastive:
        assert not args.use_clip
        def closed_set_contrastive_forward(model, x):
            if 'efficientnet' in args.arch:
                out = model.avgpool(model.features(x)).flatten(1)
                out_norm = F.normalize(out, dim=-1)
                normalized_proj_w = F.normalize(model.classifier[1].weight, dim=-1) # [C, D]
                return torch.einsum('bd,cd->bc', out_norm, normalized_proj_w) / args.temperature
            elif 'resnet' in args.arch:
                out = model._forward_impl(x, with_fc=False)
                out_norm = F.normalize(out, dim=-1)
                normalized_proj_w = F.normalize(model.fc.weight, dim=-1) # [C, D]
                return torch.einsum('bd,cd->bc', out_norm, normalized_proj_w) / args.temperature
            else:
                raise NotImplementedError()
        model.forward = lambda x: closed_set_contrastive_forward(model, x)
        
        
    if args.use_clip and args.clip_align_image_contrastive_prototype:
        model.register_buffer('prototype', torch.zeros([len(train_dataset.class_to_idx.keys()), clip_feats_dim]))
        model.register_buffer('prototype_running_mean', torch.zeros(len(train_dataset.class_to_idx.keys())))
    if args.clip_align_image_global_bias or args.clip_align_image_global_bias_custom:
        if args.resume:
            model.register_buffer('global_bias', torch.zeros(clip_feats_dim)) # will be overwritten by checkpoint
        else:
            model.register_buffer('global_bias', train_dataset.clip_feats_mean)
    if args.clip_align_image_global_bias_custom:
        model.global_bias_custom = nn.Parameter(torch.zeros(clip_feats_dim), requires_grad=True)
    if args.clip_align_image_contrastive_projection:
        # model.img_projection = nn.Linear(clip_feats_dim, clip_feats_dim)
        model.txt_projection = nn.Linear(clip_feats_dim, clip_feats_dim)
        # model.txt_projection = nn.Sequential(
        #     nn.Linear(clip_feats_dim, clip_feats_dim), 
        #     nn.ReLU(inplace=True), 
        #     nn.Linear(clip_feats_dim, clip_feats_dim)
        # )
        # model.img_projection = nn.Sequential(
        #     nn.Linear(clip_feats_dim, clip_feats_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(clip_feats_dim * 2, clip_feats_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(clip_feats_dim * 2, clip_feats_dim),
        # )
        # model.txt_projection = nn.Sequential(
        #     nn.Linear(clip_feats_dim, clip_feats_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(clip_feats_dim * 2, clip_feats_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(clip_feats_dim * 2, clip_feats_dim),
        # )
    if args.clip_align_image_contrastive_adaptive_temperature:
        model.temp = nn.Parameter(torch.tensor(np.log(0.07)).float(), requires_grad=True)
        
    model = model.to(cuda_device)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    chatgpt_train_text_features = chatgpt_val_text_features = chatgpt_val_on_train_text_features = None
    if args.chatgpt_text_features is not None:
        chatgpt_text_features = torch.load(args.chatgpt_text_features).to(cuda_device)
        chatgpt_train_text_features = chatgpt_text_features[:len(list(train_dataset.class_to_idx.keys()))]
        chatgpt_val_on_train_text_features = chatgpt_train_text_features
        chatgpt_val_text_features = chatgpt_text_features[-len(list(val_dataset.class_to_idx.keys())):]
        
    prompt_learner = text_encoder = None
    gen_text_fxn = rand_gen_text_fxn = None
    if args.use_clip:
        label2text = {}
        chatgpt_label2text = {}
        chatgpt_lines = []
        if args.chatgpt_raw_text_file is not None:
            with open(args.chatgpt_raw_text_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 0:
                        chatgpt_lines.append(line)
        with open(args.label_path, 'r') as f:
            idx = 0
            for line in f:
                line = line.strip().split(' ')
                if len(line) > 0:
                    line[2] = line[2].replace('_', ' ')
                    label2text[line[0]] = line[2]
                    if args.chatgpt_raw_text_file is not None:
                        chatgpt_label2text[line[0]] = chatgpt_lines[idx]
                    idx += 1
        if args.chatgpt_raw_text_file is not None:
            assert len(list(label2text.keys())) == len(chatgpt_lines), f"{len(label2text.keys())} != {len(chatgpt_lines)}"
        
        print("train class_to_idx", train_dataset.class_to_idx)
        print("test class_to_idx", val_dataset.class_to_idx)
        
        if args.chatgpt_raw_text_file is not None:
            gen_text_fxn = lambda x: label2text[x] + " . " + chatgpt_label2text[x]
            def rand_drop_chatgpt_text(x):
                separated = chatgpt_label2text[x].split(' ')
                result = []
                for tok in separated:
                    if random.random() < 0.8:
                        result.append(tok)
                return ' '.join(result)
            def rand_gen_text_fxn():
                dropped_text = ["a photo of " + label2text[x] + " . " + rand_drop_chatgpt_text(x) for x in train_dataset.class_to_idx.keys()]
                print(dropped_text)
                return clip_model.encode_text(clip.tokenize(dropped_text, truncate=True).to(clip_device)).float().detach()
        else:
            gen_text_fxn = lambda x: label2text[x]
            rand_gen_text_fxn = None
        
        train_text_labels = ["a photo of " + gen_text_fxn(x) for x in train_dataset.class_to_idx.keys()]
        val_text_labels = ["a photo of " + gen_text_fxn(x) for x in val_dataset.class_to_idx.keys()]
        if val_on_train_dataset is not None:
            val_on_train_text_labels = ["a photo of " + gen_text_fxn(x) for x in val_on_train_dataset.class_to_idx.keys()]
        else:
            val_on_train_text_labels = None
            
        if args.prompt_learner:
            from models.misc.prompt_learner import PromptLearner, TextEncoder
            prompt_learner = PromptLearner(clip_model, [gen_text_fxn(x) for x in train_dataset.class_to_idx.keys()], \
                                            [gen_text_fxn(x) for x in val_dataset.class_to_idx.keys()], 
                                            [gen_text_fxn(x) for x in val_on_train_dataset.class_to_idx.keys()] if val_on_train_dataset is not None else None,
                                            device=cuda_device,
                                            cocoop=False,
                                            n_ctx=args.prompt_learner_nctx,)
            text_encoder = TextEncoder(clip_model)
        else:
            prompt_learner = text_encoder = None
            
        print("train_text_labels", train_text_labels)
        print("val_text_labels", val_text_labels)
        
        if args.ood_text is not None:
            ood_text_labels = set([x.strip() for x in open(args.ood_text, 'r').readlines()])
            ood_text_labels = set(["a photo of " + ood_text_label.replace('_', ' ') for ood_text_label in ood_text_labels])
            ood_text_labels = ood_text_labels - set(train_text_labels) - set(val_text_labels)
            print("ood_text_labels", ood_text_labels)
            
        if args.clip_repo == 'clip': 
            train_text_features = clip_model.encode_text(clip.tokenize(train_text_labels, truncate=True).to(clip_device)).float().detach()
            val_text_features = clip_model.encode_text(clip.tokenize(val_text_labels, truncate=True).to(clip_device)).float().detach()
            if val_on_train_text_labels is not None:
                val_on_train_text_features = clip_model.encode_text(clip.tokenize(val_on_train_text_labels, truncate=True).to(clip_device)).float().detach()
            else:
                val_on_train_text_features = None
            ood_text_features = clip_model.encode_text(clip.tokenize(ood_text_labels, truncate=True).to(clip_device)).float().detach() \
                                if args.ood_text is not None else None
        elif args.clip_repo == 'open_clip':
            tokenize = open_clip.tokenizer.tokenize
            train_text_features = clip_model.encode_text(tokenize(train_text_labels).to(clip_device)).float().detach()
            val_text_features = clip_model.encode_text(tokenize(val_text_labels).to(clip_device)).float().detach()
            if val_on_train_text_labels is not None:
                val_on_train_text_features = clip_model.encode_text(tokenize(val_on_train_text_labels).to(clip_device)).float().detach()
            else:
                val_on_train_text_features = None
            ood_text_features = clip_model.encode_text(clip.tokenize(ood_text_labels).to(clip_device)).float().detach() \
                                if args.ood_text is not None else None
                                
        if args.ood_text is not None:
            # auxiliary out of distribution labels
            # relevant_norm = nn.functional.normalize(torch.cat([train_text_features, val_text_features]), dim=1)
            # train text features already contain val text features
            ood_norm = nn.functional.normalize(ood_text_features, dim=-1)
            relevant_train_norm = nn.functional.normalize(train_text_features, dim=-1)
            relevant_train_scores = torch.einsum('ni,mi->nm', relevant_train_norm, ood_norm)
            values_train, ids_train = relevant_train_scores.sort(dim=-1)
            ids_train_chosen = ids_train[:, :-int(ids_train.shape[1] * 0.15)]
            ood_text_features = ood_text_features[ids_train_chosen] # [num_train_classes, num_selected_ood_classes, d]
            print("ood text features shape", ood_text_features.shape)
            
            # if args.few_shot_num > 0 and args.few_shot_method == 'finetune':
            #     relevant_val_norm = nn.functional.normalize(val_text_features, dim=1)
            #     relevant_val_scores = torch.einsum('ni,mi->nm', relevant_val_norm, ood_norm)
            #     values_val, ids_val = relevant_val_scores.sort(dim=-1)
            #     ids_val_chosen = ids_val[:, :-int(ids_val.shape[1] * 0.15)]
            #     ood_val_text_features = ood_text_features[ids_val_chosen] # [num_val_classes, num_selected_ood_classes, d]
            #     ood_text_features = torch.cat([ood_train_text_features, ood_val_text_features], dim=0)
            # else:
            #     ood_text_features = ood_train_text_features
    else:
        train_text_features = val_text_features = val_on_train_text_features = ood_text_features = None
        
    if args.few_shot_num > 0 and args.few_shot_method == 'retrieval':
        # construct few shot samples
        few_shot_features = {}
        support_set_idx = []
        assert args.use_clip
        
        # follows TIP-Adapter
        num_augs = 10
        aug_transform = transforms.Compose([
            transforms.RandAugment(2, 9),
        ])
        tmp_dataset = datasets.ImageFolder(valdir)
        for idx, (img, target) in enumerate(tmp_dataset):
            if target not in few_shot_features.keys():
                few_shot_features[target] = []
            if len(few_shot_features[target]) < args.few_shot_num:
                support_set_idx.append(idx)
                imgs = [img]
                for _ in range(num_augs - 1):
                    imgs.append(aug_transform(img))
                few_shot_features[target].extend(imgs)
                
        for k in few_shot_features.keys():
            inp = torch.tensor(np.stack([clip_preprocess_orig(x) for x in few_shot_features[k]])).to(clip_device)
            with torch.no_grad():
                out = clip_model.encode_image(inp).float()
            if cuda_device != clip_device:
                out = out.to(cuda_device)
            few_shot_features[k] = F.normalize(out, dim=-1)
        few_shot_features = torch.stack(list(few_shot_features.values())) # (num_classes, few_shot_num, dim)
        support_set_idx = torch.tensor(support_set_idx)
    else:
        few_shot_features = None
        support_set_idx = None

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    
    params_to_optimize = list(model.parameters())
    if args.use_clip and prompt_learner is not None:
        for name, param in prompt_learner.named_parameters():
            if param.requires_grad:
                params_to_optimize.append(param)
                # print("Prompt learner param to optimize:", name)
    if not args.use_adam:
        optimizer = optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=0.01) # hardcoded
    scheduler = None
    if args.onecyclelr:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=1, epochs=args.epochs)

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print(f'==> Resuming from checkpoint.. {args.resume}')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        # args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location=cuda_device)
        best_acc = checkpoint['best_acc']
        if args.few_shot_method != 'finetune':
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = 0
        
        pth1 = Path(os.path.join(args.resume, 'checkpoint.pth.tar')).parent.absolute()
        pth2 = Path(os.path.join(args.checkpoint, 'log.txt')).parent.absolute()
        if pth1 == pth2:
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=False)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Val On Train Acc.'])
        
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
        if len(missing_keys) > 0:
            print("Missing model keys:", missing_keys)
        if len(unexpected_keys) > 0:
            print("Unexpected model keys:", unexpected_keys)
        if args.few_shot_method != 'finetune':
            optimizer.load_state_dict(checkpoint['optimizer'])
        if args.onecyclelr and args.few_shot_method != 'finetune':
            scheduler.load_state_dict(checkpoint['scheduler'])
            
        if prompt_learner is not None:
            if 'prompt_learner' in checkpoint.keys():
                prompt_learner_state_dict = checkpoint['prompt_learner']
                keys = list(prompt_learner_state_dict.keys())
                for k in keys:
                    if 'token_prefix' in k or 'token_suffix' in k:
                        prompt_learner_state_dict.pop(k)
                prompt_learner.load_state_dict(checkpoint['prompt_learner'], strict=False)
            else:
                print("No prompt learner in checkpoint, initializing prompt learner from scratch...")
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Val On Train Acc.'])
    
    avg_logger = Logger(os.path.join(args.checkpoint, 'log_avg.txt'), title=title)
    avg_logger.set_names(['Avg Train Loss.', 'Avg Val Loss.', 'Avg Train Acc.', 'Avg Val Acc.', 'Avg Val On Train Acc.'])
    avg_train_loss = AverageMeter()
    avg_val_loss = AverageMeter()
    avg_train_acc = AverageMeter() 
    avg_val_acc = AverageMeter()
    avg_val_on_train_acc = AverageMeter()

    if args.evaluate:
        eval_log_path = os.path.join(args.checkpoint, 'log_eval.txt')
        resume_eval_log = os.path.exists(eval_log_path)
        eval_logger = Logger(eval_log_path, title=title, resume=resume_eval_log)
        eval_logger.set_names(['val macc', 'val on train macc'])
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda, val_transform, 
                                   val_text_features, clip_model, clip_preprocess, clip_device,
                                   few_shot_features=few_shot_features, support_set_idx=support_set_idx,
                                   chatgpt_text_features=chatgpt_val_text_features if args.chatgpt_text_features is not None else None,
                                   chatgpt_ensemble=args.chatgpt_ensemble,
                                   prompt_learner=prompt_learner, text_encoder=text_encoder, prompt_mode='test')
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc), flush=True)
        if val_on_train_loader is not None:
            val_on_train_loss, val_on_train_acc = test(val_on_train_loader, model, criterion, start_epoch, use_cuda, val_transform,
                                        val_on_train_text_features, clip_model, clip_preprocess, clip_device,
                                        few_shot_features=few_shot_features, support_set_idx=support_set_idx,
                                        chatgpt_text_features=chatgpt_val_on_train_text_features if args.chatgpt_text_features is not None else None,
                                        chatgpt_ensemble=args.chatgpt_ensemble,
                                        prompt_learner=prompt_learner, text_encoder=text_encoder, prompt_mode='val_on_train')
            print(' Val on Train Loss:  %.8f, Val on Train Acc:  %.2f' % (val_on_train_loss, val_on_train_acc), flush=True)
        eval_logger.append([test_acc, val_on_train_acc])
        eval_logger.close()
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, scheduler)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        
        for _ in range(args.repeat_epochs):
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.epochs, use_cuda, train_transform, 
                                        train_text_features, ood_text_features, clip_model, clip_preprocess, clip_device,
                                        prompt_learner=prompt_learner,
                                        text_encoder=text_encoder,
                                        prompt_mode='train',
                                        clip_align_image_classification=args.clip_align_image_classification,
                                        clip_align_image_aux_caption=args.clip_align_image_aux_caption,
                                        chatgpt_text_features=chatgpt_train_text_features,
                                        chatgpt_ensemble=args.chatgpt_ensemble,
                                        rand_drop_text=args.rand_drop_text,
                                        rand_gen_text_fxn=rand_gen_text_fxn,
                                        clip_align_proximal_image_num=args.clip_align_proximal_image_num,
                                        clip_align_proximal_text_num=args.clip_align_proximal_text_num,
                                        clip_align_proximal_text_axis=args.clip_align_proximal_text_axis,
                                        clip_align_text_only_proximal=args.clip_align_text_only_proximal,
                                        clip_filter_out_wrong_alignment=args.clip_filter_out_wrong_alignment,
                                        clip_align_image_mse=args.clip_align_image_mse,
                                        clip_align_image_mse_unnorm=args.clip_align_image_mse_unnorm,
                                        clip_align_image_contrastive=args.clip_align_image_contrastive,
                                        clip_align_image_contrastive_mode=args.clip_align_image_contrastive_mode,
                                        clip_align_image_contrastive_only_other=args.clip_align_image_contrastive_only_other,
                                        clip_align_image_contrastive_random_combine=args.clip_align_image_contrastive_random_combine,
                                        clip_align_image_contrastive_hard_sample=args.clip_align_image_contrastive_hard_sample,
                                        clip_align_image_contrastive_prototype=args.clip_align_image_contrastive_prototype,
                                        clip_align_image_contrastive_projection=args.clip_align_image_contrastive_projection,
                                        clip_align_image_contrastive_adaptive_temperature=args.clip_align_image_contrastive_adaptive_temperature,
                                        clip_align_image_relative_vector=args.clip_align_image_relative_vector,
                                        clip_align_text_hinge=args.clip_align_text_hinge,)
        if not args.skip_val:
            test_target_remap = None
            if clip_model is None:
                train_n_cls = len(train_dataset.class_to_idx.keys())
                val_n_cls = len(val_dataset.class_to_idx.keys())
                if args.few_shot_num > 0:
                    test_target_remap = [train_n_cls - val_n_cls, train_n_cls]
                else:
                    test_target_remap = [train_n_cls, train_n_cls + val_n_cls]
            test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda, val_transform, 
                                        val_text_features, clip_model, clip_preprocess, clip_device,
                                        chatgpt_text_features=chatgpt_val_text_features,
                                        chatgpt_ensemble=args.chatgpt_ensemble,
                                        target_remap=test_target_remap,
                                        few_shot_features=few_shot_features, support_set_idx=support_set_idx,
                                        prompt_learner=prompt_learner, text_encoder=text_encoder, prompt_mode='test')
            val_on_train_acc = 0.0
            if (args.epochs - epoch - 1 < 5) and (val_on_train_loader is not None):
                # in distribution accuracy
                _, val_on_train_acc = test(val_on_train_loader, model, criterion, epoch, use_cuda, val_transform,
                                        val_on_train_text_features, clip_model, clip_preprocess, clip_device,
                                        chatgpt_text_features=chatgpt_val_on_train_text_features,
                                        chatgpt_ensemble=args.chatgpt_ensemble,
                                        prompt_learner=prompt_learner, text_encoder=text_encoder, prompt_mode='val_on_train')
            # append logger file
            logger.append([optimizer.param_groups[0]['lr'], train_loss, test_loss, train_acc, test_acc, val_on_train_acc])
            
            if args.epochs - epoch <= 5:
                avg_train_loss.update(train_loss)
                avg_val_loss.update(test_loss)
                avg_train_acc.update(train_acc)
                avg_val_acc.update(test_acc)
                avg_val_on_train_acc.update(val_on_train_acc)
                if args.epochs - epoch == 1:
                    avg_logger.append([avg_train_loss.avg, avg_val_loss.avg, avg_train_acc.avg, avg_val_acc.avg, avg_val_on_train_acc.avg])
                    avg_logger.close()       

            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_dict = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict() if scheduler is not None else None,
                }
            if prompt_learner is not None:
                save_dict['prompt_learner'] = prompt_learner.state_dict()
            save_checkpoint(save_dict, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, total_epochs, use_cuda, 
          train_transform, 
          train_text_features=None, ood_text_features=None, clip_model=None, clip_preprocess=None, clip_device=None,
          prompt_learner=None,
          text_encoder=None,
          prompt_mode='train',
          clip_align_image_classification=False, 
          clip_align_image_aux_caption=False,
          chatgpt_text_features=None,
          chatgpt_ensemble=False,
          rand_drop_text=False,
          rand_gen_text_fxn=None,
          clip_align_proximal_image_num=-1,
          clip_align_proximal_text_num=-1,
          clip_align_proximal_text_axis=1,
          clip_align_text_only_proximal=False,
          clip_filter_out_wrong_alignment=False,
          clip_align_image_mse=False, clip_align_image_mse_unnorm=False,
          clip_align_image_contrastive=False,
          clip_align_image_contrastive_mode='single',
          clip_align_image_contrastive_only_other=False,
          clip_align_image_contrastive_random_combine=False,
          clip_align_image_contrastive_hard_sample=False,
          clip_align_image_contrastive_prototype=False,
          clip_align_image_contrastive_projection=False,
          clip_align_image_contrastive_adaptive_temperature=False,
          clip_align_image_relative_vector=None,
          clip_align_text_hinge=False,):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    losses = AverageMeter()
    aux_mse_losses = AverageMeter()
    aux_contrastive_losses = AverageMeter()
    aux_contrastive_prototype_losses = AverageMeter()
    aux_relative_vector_losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    
    avg_accuracy_per_class = None
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if clip_model is not None:
            inputs, outputs_clip = inputs
        
        if clip_align_image_aux_caption:
            outputs_clip, aux_clip_caption = outputs_clip
        else:
            aux_clip_caption = None
        
        inputs = train_transform(inputs)
        
        if use_cuda:
            inputs, targets = inputs.to(cuda_device), targets.to(cuda_device)
        if clip_model is not None:
            outputs_clip = outputs_clip.to(cuda_device)
        if aux_clip_caption is not None:
            aux_clip_caption = aux_clip_caption.to(cuda_device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        if hasattr(model, 'global_bias'):
            outputs = outputs + model.global_bias[None, :]
        if hasattr(model, 'global_bias_custom'):
            outputs = outputs + model.global_bias_custom[None, :]
        loss = None
        aux_mse_loss = None
        aux_contrastive_loss = None
        aux_contrastive_prototype_loss = None
        aux_relative_vector_loss = None
        if clip_model is not None:
            outputs_norm = nn.functional.normalize(outputs, dim=-1)
            ## input: normalized image features from distilled network, output: unnormalized clip text features
            if prompt_learner is not None and text_encoder is not None:
                assert prompt_mode == 'train'
                prompts = prompt_learner(outputs_norm)
                train_text_features = [] # override the input train_text_features since the prompt is adaptive
                for pts_i in prompts: # (n_cls, n_tkn, ctx_dim)
                    minib = 128
                    tokenized_idx = 0
                    cur_train_text_feature = []
                    while tokenized_idx < len(prompt_learner.train_tokenized_prompts):
                        cur_train_text_feature.append(
                            text_encoder(pts_i[tokenized_idx:tokenized_idx+minib], prompt_learner.train_tokenized_prompts[tokenized_idx:tokenized_idx+minib])
                        )
                        tokenized_idx += minib
                    cur_train_text_feature = torch.cat(cur_train_text_feature, dim=0)
                    train_text_features.append(cur_train_text_feature[None, ...])
                train_text_features = torch.cat(train_text_features, dim=0) # (batch_size, n_cls, dim) or (1, n_cls, dim) depending on whether prompt_learner mode is 'dual' or 'single'
            elif rand_drop_text:
                assert rand_gen_text_fxn is not None
                train_text_features = rand_gen_text_fxn()
                
            if chatgpt_text_features is not None:
                if not chatgpt_ensemble:
                    cur_train_text_features = chatgpt_text_features[:, -1, :] # override the input train_text_features with chatgpt_text_features
                else:
                    cur_train_text_features = chatgpt_text_features
            else:
                cur_train_text_features = train_text_features
            if clip_align_image_contrastive_projection:
                # outputs_clip = model.img_projection(outputs_clip)
                cur_train_text_features = model.txt_projection(cur_train_text_features)
            cur_train_text_features_norm = nn.functional.normalize(cur_train_text_features, dim=-1)
            if cur_train_text_features_norm.dim() == 3:
                if chatgpt_text_features is None or not chatgpt_ensemble:
                    # batch dependent
                    classify_outputs = torch.einsum('ni,nci->nc', outputs_norm, cur_train_text_features_norm)
                else:
                    classify_outputs = torch.einsum('ni,mci->nmc', outputs_norm, cur_train_text_features_norm)
            elif cur_train_text_features_norm.dim() == 2:
                classify_outputs = torch.einsum('ni,ci->nc', outputs_norm, cur_train_text_features_norm)
            else:
                raise NotImplementedError
                    
            outputs_clip_norm = nn.functional.normalize(outputs_clip, dim=-1)
            
            if aux_clip_caption is not None:
                if clip_align_image_contrastive_projection:
                    aux_clip_caption = model.txt_projection(aux_clip_caption)
                aux_clip_caption_norm = nn.functional.normalize(aux_clip_caption, dim=-1)
            
            if clip_align_image_classification:
                if not clip_align_text_hinge:
                    if ood_text_features is not None:
                        if ood_text_features.dim() == 3:
                            cur_ood_text_features = ood_text_features[targets]
                            if clip_align_image_contrastive_projection:
                                cur_ood_text_features = model.txt_projection(cur_ood_text_features)
                            cur_ood_text_features_norm = nn.functional.normalize(cur_ood_text_features, dim=-1)
                            classify_outputs_ood = torch.einsum('ni,nci->nc', outputs_norm, cur_ood_text_features_norm)
                        elif ood_text_features.dim() == 2:
                            cur_ood_text_features = ood_text_features
                            if clip_align_image_contrastive_projection:
                                cur_ood_text_features = model.txt_projection(cur_ood_text_features)
                            cur_ood_text_features_norm = nn.functional.normalize(cur_ood_text_features, dim=-1)
                            classify_outputs_ood = torch.einsum('ni,ci->nc', outputs_norm, cur_ood_text_features_norm)
                        else:
                            raise NotImplementedError()
                        classify_outputs = torch.cat([classify_outputs, classify_outputs_ood], dim=1)
                    classify_outputs = classify_outputs / args.temperature # temperature
                    if classify_outputs.ndim == 2:
                        loss = criterion(classify_outputs, targets)
                    elif classify_outputs.ndim == 3:
                        loss = criterion(classify_outputs, targets[:, None].tile(1, classify_outputs.shape[-1]))
                        
                    if clip_align_proximal_text_num > 0:
                        if cur_train_text_features_norm.dim() == 3:
                            if chatgpt_text_features is None or not chatgpt_ensemble:
                                # batch dependent
                                classify_outputs_clip = torch.einsum('ni,nci->nc', outputs_clip_norm, cur_train_text_features_norm)
                            else:
                                classify_outputs_clip = torch.einsum('ni,mci->nmc', outputs_clip_norm, cur_train_text_features_norm)
                        elif cur_train_text_features_norm.dim() == 2:
                            classify_outputs_clip = torch.einsum('ni,ci->nc', outputs_clip_norm, cur_train_text_features_norm)
                            
                        classify_outputs_for_align = classify_outputs
                        classify_outputs_clip_for_align = classify_outputs_clip
                        targets_for_align = targets if classify_outputs_clip.ndim == 2 else targets[:, None]
                        if clip_filter_out_wrong_alignment:
                            clip_correct_bool = (classify_outputs_clip_for_align.argmax(dim=-1) == targets_for_align)
                            classify_outputs_for_align = classify_outputs_for_align[clip_correct_bool] # [N', C]
                            classify_outputs_clip_for_align = classify_outputs_clip_for_align[clip_correct_bool]
                            
                        classify_outputs_clip_for_align = classify_outputs_clip_for_align / args.temperature # temperature
                        
                        if clip_align_text_only_proximal: # only use the KL between CLIP logits for the proximal text and predicted logits as the classification loss 
                            loss = 0.0
                            
                        if clip_align_proximal_text_axis == 1:
                            classify_outputs_clip_for_align_topk_values, classify_outputs_clip_for_align_topk_ids = classify_outputs_clip_for_align.topk(
                                k=min(clip_align_proximal_text_num, classify_outputs_clip_for_align.shape[-1]), dim=-1) # [N, K] or [N, M, K]
                            classify_outputs_for_align_topk_values = classify_outputs_for_align.gather(-1, classify_outputs_clip_for_align_topk_ids) # [N, K] or [N, M, K]
                            # print(classify_outputs_clip_for_align_topk_values[:3], classify_outputs_for_align_topk_values[:3])
                            # print(classify_outputs_clip_for_align_topk_values[:3].softmax(dim=-1), classify_outputs_for_align_topk_values[:3].softmax(dim=-1))
                            loss = loss + 1.0 * (F.softmax(classify_outputs_clip_for_align_topk_values, dim=-1) 
                                        * (F.log_softmax(classify_outputs_clip_for_align_topk_values, dim=-1) - F.log_softmax(classify_outputs_for_align_topk_values, dim=-1)
                                        )).sum(dim=-1).mean()
                        else:
                            assert clip_align_proximal_text_axis == 0
                            classify_outputs_clip_for_align_topk_values, classify_outputs_clip_for_align_topk_ids = classify_outputs_clip_for_align.topk(
                                k=min(clip_align_proximal_text_num, classify_outputs_clip_for_align.shape[0]), dim=0) # [K, C] or [K, M, C]
                            classify_outputs_for_align_topk_values = classify_outputs_for_align.gather(0, classify_outputs_clip_for_align_topk_ids) # [K, C] or [K, M, C]
                            # print(classify_outputs_clip_topk_values[..., :3], classify_outputs_topk_values[..., :3])
                            # print(classify_outputs_clip_for_align_topk_values.softmax(dim=0)[:3], classify_outputs_for_align_topk_values.softmax(dim=0)[:3])
                            loss = loss + 1.0 * (F.softmax(classify_outputs_clip_for_align_topk_values, dim=0) 
                                        * (F.log_softmax(classify_outputs_clip_for_align_topk_values, dim=0) - F.log_softmax(classify_outputs_for_align_topk_values, dim=0)
                                        )).sum(dim=0).mean()
                            # print((F.softmax(classify_outputs_clip_for_align_topk_values, dim=0) 
                            #             * (F.log_softmax(classify_outputs_clip_for_align_topk_values, dim=0) - F.log_softmax(classify_outputs_for_align_topk_values, dim=0)
                            #             )).sum(dim=0).mean())
                        
                        
                        
                        # classify_outputs_m_target = classify_outputs.clone()
                        # classify_outputs_m_target[torch.arange(targets.shape[0], device=targets.device), targets] = -1e9
                        # loss = loss - 1.0 * (classify_outputs_topk_values[..., 1:].mean(dim=-1) - classify_outputs_m_target.logsumexp(dim=-1)).mean()
                        
                        # loss = loss + (classify_outputs_clip_topk_values - classify_outputs_topk_values).abs().mean()
                        
                        # print('aux txt loss', (F.softmax(classify_outputs_clip_topk_values, dim=-1) 
                        #                 * (F.log_softmax(classify_outputs_clip_topk_values, dim=-1) - F.log_softmax(classify_outputs_topk_values, dim=-1)
                        #                 )).sum(dim=-1).mean())
                        # print((classify_outputs_clip_topk_values - classify_outputs_topk_values).abs().mean())
                else:
                    classify_outputs_for_hinge = classify_outputs - classify_outputs[torch.arange(targets.shape[0], device=targets.device), targets][:, None]
                    loss = torch.clamp(classify_outputs_for_hinge + 0.2, min=0.0).sum(dim=-1).mean() # hardcoded
                    
                if clip_align_image_aux_caption:
                    aux_classify_outputs = torch.einsum('ni,ci->nc', outputs_norm, aux_clip_caption_norm)
                    contrastive_invalid_matrix = (targets[:, None] == targets[None, :])
                    tmp = torch.arange(contrastive_invalid_matrix.size(0), device=contrastive_invalid_matrix.device)
                    contrastive_invalid_matrix[tmp, tmp] = False
                    aux_classify_outputs[contrastive_invalid_matrix] = -1e7
                    arange = torch.arange(aux_classify_outputs.shape[0], device=aux_classify_outputs.device)
                    loss += criterion(aux_classify_outputs, arange)
                    # loss += 0.5 * (criterion(aux_classify_outputs, arange) + criterion(aux_classify_outputs.T, arange))
                
            if (clip_align_image_mse or clip_align_image_mse_unnorm 
                or clip_align_image_contrastive or clip_align_image_contrastive_only_other or clip_align_image_contrastive_random_combine
                or clip_align_image_contrastive_hard_sample
                or clip_align_image_contrastive_projection or clip_align_image_contrastive_prototype):
                
                # assert cur_train_text_features_norm.dim() == 2, "Not Implemented"
                # This hurts performance for some reason, so we don't filter out wrong alignments
                # if clip_filter_out_wrong_alignment:
                #     classify_outputs_clip = torch.einsum('ni,ci->nc', outputs_clip_norm, cur_train_text_features_norm)
                #     clip_correct_bool = (classify_outputs_clip.argmax(dim=-1) == targets)
                # else:
                #     clip_correct_bool = torch.ones_like(targets, dtype=torch.bool)
                clip_correct_bool = torch.ones_like(targets, dtype=torch.bool)
                # print(clip_correct_bool.sum())
                            
                if not clip_align_image_contrastive_adaptive_temperature:
                    cur_temp = args.temperature
                else:
                    # temp_init = args.temperature
                    # temp_end = 10.0
                    # temp_init_log = np.log(temp_init)
                    # temp_end_log = np.log(temp_end)
                    # cur_temp = np.exp(temp_init_log + (temp_end_log - temp_init_log) * epoch / total_epochs)
                    cur_temp = torch.clamp(torch.exp(model.temp), min=0.01)
                if clip_align_image_mse:
                    aux_mse_loss = ((outputs_norm - outputs_clip_norm) ** 2).sum(dim=-1)[clip_correct_bool].mean()
                if clip_align_image_mse_unnorm:
                    aux_mse_loss = torch.log(1 + ((outputs - outputs_clip) ** 2).sum(dim=-1)[clip_correct_bool].mean() / 10.0)
                    # aux_mse_loss = torch.abs(outputs - outputs_clip).sum(dim=-1).mean()
                if (clip_align_image_contrastive or clip_align_image_contrastive_only_other or clip_align_image_contrastive_random_combine
                    or clip_align_image_contrastive_hard_sample
                    or clip_align_image_contrastive_projection):
                    contrastive_mat = torch.einsum('ni,mi->nm', outputs_norm[clip_correct_bool], outputs_clip_norm[clip_correct_bool]) / cur_temp
                    contrastive_labels = torch.arange(contrastive_mat.size(0), device=contrastive_mat.device)
                    if clip_align_image_contrastive_hard_sample:
                        # contrastive_invalid_matrix = (contrastive_mat < 0.03) # hardcoded
                        contrastive_invalid_matrix = torch.ones_like(contrastive_mat).bool() # hardcoded
                        thres = int(0.3 * contrastive_mat.size(0))
                        contrastive_invalid_matrix[contrastive_mat.sort(dim=-1).indices[:, -thres:]] = False
                        tmp = torch.arange(contrastive_invalid_matrix.size(0), device=contrastive_invalid_matrix.device)
                        contrastive_invalid_matrix[tmp, tmp] = False
                        contrastive_mat[contrastive_invalid_matrix] = -1e7
                    if clip_align_image_contrastive_only_other:
                        contrastive_invalid_matrix = (targets[clip_correct_bool, None] == targets[None, clip_correct_bool])
                        tmp = torch.arange(contrastive_invalid_matrix.size(0), device=contrastive_invalid_matrix.device)
                        contrastive_invalid_matrix[tmp, tmp] = False
                        contrastive_mat[contrastive_invalid_matrix] = -1e7
                        
                    if clip_align_image_contrastive_mode == 'single':
                        aux_contrastive_loss = F.cross_entropy(contrastive_mat, contrastive_labels)
                    elif clip_align_image_contrastive_mode == 'bidirectional':
                        aux_contrastive_loss = 0.5 * (
                            F.cross_entropy(contrastive_mat, contrastive_labels)
                            + F.cross_entropy(contrastive_mat.T, contrastive_labels)
                        )
                    elif clip_align_image_contrastive_mode == 'kl':
                        aux_contrastive_loss = 0.5 * (
                            F.cross_entropy(contrastive_mat, contrastive_labels)
                            + F.cross_entropy(contrastive_mat.T, contrastive_labels)
                        )
                        contrastive_mat_clip = torch.einsum('ni,mi->nm', outputs_clip_norm[clip_correct_bool], outputs_clip_norm[clip_correct_bool]) / cur_temp
                        k = min(10, contrastive_mat_clip.shape[-1])
                        contrastive_mat_clip_topk_values, contrastive_mat_clip_topk_ids = contrastive_mat_clip.topk(k=k, dim=-1) # [N, K]
                        contrastive_mat_topk_values = contrastive_mat.gather(-1, contrastive_mat_clip_topk_ids) # [N, K]
                        aux_contrastive_loss += 2.5 * (F.softmax(contrastive_mat_clip_topk_values, dim=-1) 
                                        * (F.log_softmax(contrastive_mat_clip_topk_values, dim=-1) - F.log_softmax(contrastive_mat_topk_values, dim=-1)
                                        )).sum(dim=-1).mean()
                        contrastive_mat_topk_values = contrastive_mat.gather(0, contrastive_mat_clip_topk_ids.T).T # [N, K]
                        aux_contrastive_loss += 2.5 * (F.softmax(contrastive_mat_clip_topk_values, dim=-1) 
                                        * (F.log_softmax(contrastive_mat_clip_topk_values, dim=-1) - F.log_softmax(contrastive_mat_topk_values, dim=-1)
                                        )).sum(dim=-1).mean()
                    else:
                        raise NotImplementedError()
                    
                    if clip_align_proximal_image_num > 0:
                        contrastive_mat_self = torch.einsum('ni,mi->nm', outputs_norm[clip_correct_bool], outputs_norm[clip_correct_bool]) / cur_temp
                        contrastive_mat_clip = torch.einsum('ni,mi->nm', outputs_clip_norm[clip_correct_bool], outputs_clip_norm[clip_correct_bool]) / cur_temp
                        contrastive_mat_clip_topk_values, contrastive_mat_clip_topk_ids = contrastive_mat_clip.topk(
                            k=min(clip_align_proximal_image_num, contrastive_mat_clip.shape[-1]), dim=-1) # [N, K]
                        contrastive_mat_self_topk_values = contrastive_mat_self.gather(-1, contrastive_mat_clip_topk_ids) # [N, K]
                        aux_contrastive_loss = aux_contrastive_loss + 1.0 * (F.softmax(contrastive_mat_clip_topk_values, dim=-1) 
                                        * (F.log_softmax(contrastive_mat_clip_topk_values, dim=-1) - F.log_softmax(contrastive_mat_self_topk_values, dim=-1)
                                        )).sum(dim=-1).mean()
                        
                        # contrastive_mat_self_m_target = contrastive_mat_self.clone()
                        # arange = torch.arange(contrastive_mat_self.shape[0], device=contrastive_mat_self.device)
                        # contrastive_mat_self_m_target[arange, arange] = -1e9
                        # aux_contrastive_loss = aux_contrastive_loss - 1.0 * (contrastive_mat_self_topk_values[..., 1:].mean(dim=-1) - contrastive_mat_self_m_target.logsumexp(dim=-1)).mean()
                        
                        print(contrastive_mat_clip_topk_values[:3], contrastive_mat_self_topk_values[:3])
                        print(contrastive_mat_clip_topk_values[:3].softmax(dim=-1), contrastive_mat_self_topk_values[:3].softmax(dim=-1))
                        print("aux img loss", (F.softmax(contrastive_mat_clip_topk_values, dim=-1) 
                                        * (F.log_softmax(contrastive_mat_clip_topk_values, dim=-1) - F.log_softmax(contrastive_mat_self_topk_values, dim=-1)
                                        )).sum(dim=-1).mean())
                        # print((F.softmax(contrastive_mat_clip_topk_values, dim=-1) 
                        #                 * (F.log_softmax(contrastive_mat_clip_topk_values, dim=-1) - F.log_softmax(contrastive_mat_self_topk_values, dim=-1)
                        #                 ))[0])
                        # print("neigh repulse loss", -(contrastive_mat_self_topk_values[..., 1:].mean(dim=-1) - contrastive_mat_self_m_target.logsumexp(dim=-1)).mean())
                        # print(contrastive_mat_self_m_target[0])
                        
                    # if clip_align_image_contrastive_adaptive_temperature:
                    #     aux_contrastive_loss *= cur_temp
                    
                    if clip_align_image_contrastive_random_combine:
                        rand_weight = torch.randn(outputs_clip[clip_correct_bool].shape[0] * 2, outputs_clip[clip_correct_bool].shape[0], device=outputs_clip.device)
                        rand_weight = torch.softmax(rand_weight / 0.5, dim=1)
                        outputs_clip_rand_comb = torch.einsum('nm,mi->ni', rand_weight, outputs_clip[clip_correct_bool])
                        outputs_clip_rand_comb_norm = nn.functional.normalize(outputs_clip_rand_comb, dim=1)
                        # contrastive_mat = torch.cat([
                        #     torch.diag(contrastive_mat)[:, None],
                        #     torch.einsum('ni,mi->nm', outputs_norm, outputs_clip_rand_comb_norm) / cur_temp  
                        # ], dim=1)                      
                        # aux_contrastive_loss += F.cross_entropy(contrastive_mat, torch.zeros_like(contrastive_labels))                        
                        outputs_rand_comb = torch.einsum('nm,mi->ni', rand_weight, outputs[clip_correct_bool])
                        outputs_rand_comb_norm = nn.functional.normalize(outputs_rand_comb, dim=1)
                        
                        aux_contrastive_loss += (outputs_rand_comb_norm * outputs_clip_rand_comb_norm).sum(dim=-1).mean()
                        # outputs_clip_rand_comb_norm = torch.cat([outputs_clip_norm, outputs_clip_rand_comb_norm], dim=0)
                        # outputs_rand_comb_norm = torch.cat([outputs_norm, outputs_rand_comb_norm], dim=0)
                        
                        # contrastive_mat = torch.einsum('ni,mi->nm', outputs_rand_comb_norm, outputs_clip_rand_comb_norm) / cur_temp
                        # contrastive_labels = torch.arange(contrastive_mat.size(0), device=contrastive_mat.device)
                        # aux_contrastive_loss = 0.5 * (
                        #     F.cross_entropy(contrastive_mat, contrastive_labels) 
                        #     + F.cross_entropy(contrastive_mat.T, contrastive_labels)
                        # )
                        
                    if clip_align_image_contrastive_projection:
                        # contrastive loss on the projected CLIP features to keep CLIP relationships
                        assert not chatgpt_ensemble, "Not Implemented"
                        if cur_train_text_features_norm.dim() == 3:
                            contrastive_mat = torch.einsum('ni,nci->nc', outputs_clip_norm[clip_correct_bool], cur_train_text_features_norm) / cur_temp
                        elif cur_train_text_features_norm.dim() == 2:
                            contrastive_mat = torch.einsum('ni,ci->nc', outputs_clip_norm[clip_correct_bool], cur_train_text_features_norm) / cur_temp
                        else:
                            raise NotImplementedError
                        aux_contrastive_loss += F.cross_entropy(contrastive_mat, targets)
                        if aux_clip_caption is not None:
                            contrastive_mat = torch.einsum('ni,ci->nc', outputs_clip_norm[clip_correct_bool], aux_clip_caption_norm) / cur_temp
                            contrastive_invalid_matrix = (targets[clip_correct_bool, None] == targets[None, clip_correct_bool])
                            tmp = torch.arange(contrastive_invalid_matrix.size(0), device=contrastive_invalid_matrix.device)
                            contrastive_invalid_matrix[tmp, tmp] = False
                            contrastive_mat[contrastive_invalid_matrix] = -1e7
                            contrastive_labels = torch.arange(contrastive_mat.size(0), device=contrastive_mat.device)
                            aux_contrastive_loss += 0.5 * (
                                F.cross_entropy(contrastive_mat, contrastive_labels)
                                + F.cross_entropy(contrastive_mat.T, contrastive_labels)
                            )
                        
                if clip_align_image_contrastive_prototype:
                    this_n = torch.zeros_like(model.prototype[:,0])
                    this_n.scatter_(
                        dim=0, index=targets[clip_correct_bool], src=torch.ones_like(targets).float(), reduce='add'
                    )
                    this_outputs_clip = torch.zeros_like(model.prototype)
                    this_outputs_clip.scatter_(
                        dim=0, index=targets[clip_correct_bool,None].tile(1,outputs_clip[clip_correct_bool].shape[-1]), src=outputs_clip[clip_correct_bool], reduce='add'
                    )
                    this_outputs_clip = this_outputs_clip / (this_n[:,None] + 1e-6)
                    new_n = this_n + model.prototype_running_mean
                    model.prototype = (model.prototype_running_mean / (new_n + 1e-6))[:, None] * model.prototype + (this_n / (new_n + 1e-6))[:, None] * this_outputs_clip
                    model.prototype_running_mean += this_n
                    model_prototype_norm = nn.functional.normalize(model.prototype, dim=1)
                    
                    mask = (new_n[targets[clip_correct_bool]] > 0)
                    outputs_norm_masked = outputs_norm[clip_correct_bool][mask]
                    targets_masked = targets[clip_correct_bool][mask]
                    contrastive_mat = torch.einsum('ni,mi->nm', outputs_norm_masked, model_prototype_norm) / cur_temp
                    aux_contrastive_prototype_loss = F.cross_entropy(contrastive_mat, targets_masked)
                    
            if clip_align_image_relative_vector is not None:
                assert cur_train_text_features_norm.dim() == 2, "Not Implemented"
                if clip_filter_out_wrong_alignment:
                    classify_outputs_clip = torch.einsum('ni,ci->nc', outputs_clip_norm, cur_train_text_features_norm)
                    clip_correct_bool = (classify_outputs_clip.argmax(dim=-1) == targets)
                else:
                    clip_correct_bool = torch.ones_like(cur_train_text_features_norm[:, 0], dtype=torch.bool)
                    
                # interestingly normalize->subtract->w/o normalize does not help (coeff=1.0 is bad; coeff=5.0 is a bit better); 
                # normalize->subtract->normalize can help
                # outputs_pairwise_diff = outputs[:, None, :] - outputs[None, :, :]
                outputs_pairwise_diff = outputs_norm[clip_correct_bool, None, :] - outputs_norm[None, clip_correct_bool, :]
                outputs_pairwise_diff_norm = outputs_pairwise_diff / (outputs_pairwise_diff.norm(dim=-1, keepdim=True) + 1e-6)
                # outputs_clip_pairwise_diff = outputs_clip[:, None, :] - outputs_clip[None, :, :]
                outputs_clip_pairwise_diff = outputs_clip_norm[clip_correct_bool, None, :] - outputs_clip_norm[None, clip_correct_bool, :]
                outputs_clip_pairwise_diff_norm = outputs_clip_pairwise_diff / (outputs_clip_pairwise_diff.norm(dim=-1, keepdim=True) + 1e-6)   
                aux_relative_vector_loss = 0.0
                
                if "norm_diff_direction_ours" in clip_align_image_relative_vector:
                    aux_relative_vector_loss += 2.0 * (outputs_pairwise_diff_norm - outputs_clip_pairwise_diff_norm).norm(dim=-1).mean()

                if "norm_diff_ours" in clip_align_image_relative_vector:
                    aux_relative_vector_loss += 2.0 * (outputs_pairwise_diff - outputs_clip_pairwise_diff).norm(dim=-1).mean()

                if "norm_unnorm_diff_direction_ours" in clip_align_image_relative_vector:
                    tmp1 = outputs[clip_correct_bool, None, :] - outputs[None, clip_correct_bool, :]
                    tmp1 = tmp1 / (tmp1.norm(dim=-1, keepdim=True) + 1e-6)
                    tmp2 = outputs_clip[clip_correct_bool, None, :] - outputs_clip[None, clip_correct_bool, :]
                    tmp2 = tmp2 / (tmp2.norm(dim=-1, keepdim=True) + 1e-6)
                    aux_relative_vector_loss += 2.0 * (tmp1 - tmp2).norm(dim=-1).mean()

                if "norm_unnorm_diff_ours" in clip_align_image_relative_vector:
                    tmp1 = outputs[clip_correct_bool, None, :] - outputs[None, clip_correct_bool, :]
                    tmp2 = outputs_clip[clip_correct_bool, None, :] - outputs_clip[None, clip_correct_bool, :]
                    aux_relative_vector_loss += 2.0 * (tmp1 - tmp2).norm(dim=-1).mean()

                if 'norm_diff_rkd' in clip_align_image_relative_vector:
                    # aux_relative_vector_loss += 2.0 * ((outputs_pairwise_diff_norm - outputs_clip_pairwise_diff_norm) ** 2).sum(dim=-1).mean()outputs_clip_pairwise_diff
                    outputs_pairwise_diff_l2 = outputs_pairwise_diff.norm(dim=-1)
                    outputs_clip_pairwise_diff_l2 = outputs_clip_pairwise_diff.norm(dim=-1)
                    aux_relative_vector_loss += 2.0 * F.huber_loss(outputs_pairwise_diff_l2, outputs_clip_pairwise_diff_l2, delta=0.5)

                if 'norm_unnorm_diff_rkd' in clip_align_image_relative_vector:
                    # aux_relative_vector_loss += 2.0 * ((outputs_pairwise_diff_norm - outputs_clip_pairwise_diff_norm) ** 2).sum(dim=-1).mean()outputs_clip_pairwise_diff
                    outputs_pairwise_diff_l2 = (outputs[clip_correct_bool, None, :] - outputs[None, clip_correct_bool, :]).norm(dim=-1)
                    outputs_clip_pairwise_diff_l2 = (outputs_clip[clip_correct_bool, None, :] - outputs_clip[None, clip_correct_bool, :]).norm(dim=-1)
                    aux_relative_vector_loss += 2.0 * F.huber_loss(outputs_pairwise_diff_l2, outputs_clip_pairwise_diff_l2, delta=0.5)
                    
                if 'relative_angle_direction' in clip_align_image_relative_vector:
                    subsample = 2
                    outputs_pairwise_diff_norm_subsample = outputs_pairwise_diff_norm[::subsample, ::subsample]
                    outputs_clip_pairwise_diff_norm_subsample = outputs_clip_pairwise_diff_norm[::subsample, ::subsample]
                    angle_diff_pred = torch.einsum('bid,bjd->bij', outputs_pairwise_diff_norm_subsample, outputs_pairwise_diff_norm_subsample)
                    angle_diff_gt = torch.einsum('bid,bjd->bij', outputs_clip_pairwise_diff_norm_subsample, outputs_clip_pairwise_diff_norm_subsample)
                    aux_relative_vector_loss += 5.0 * (angle_diff_pred - angle_diff_gt).abs().mean()
                    
                if 'relative_angle_unnorm' in clip_align_image_relative_vector:
                    subsample = 2
                    outputs_pairwise_diff_norm_subsample = (outputs[clip_correct_bool][::subsample, None, :] - outputs[clip_correct_bool][None, ::subsample, :])
                    outputs_pairwise_diff_norm_subsample = outputs_pairwise_diff_norm_subsample / (outputs_pairwise_diff_norm_subsample.norm(dim=-1, keepdim=True) + 1e-6)
                    outputs_clip_pairwise_diff_norm_subsample = (outputs_clip[clip_correct_bool][::subsample, None, :] - outputs_clip[clip_correct_bool][None, ::subsample, :])
                    outputs_clip_pairwise_diff_norm_subsample = outputs_clip_pairwise_diff_norm_subsample / (outputs_clip_pairwise_diff_norm_subsample.norm(dim=-1, keepdim=True) + 1e-6)
                    angle_diff_pred = torch.einsum('bid,bjd->bij', outputs_pairwise_diff_norm_subsample, outputs_pairwise_diff_norm_subsample)
                    angle_diff_gt = torch.einsum('bid,bjd->bij', outputs_clip_pairwise_diff_norm_subsample, outputs_clip_pairwise_diff_norm_subsample)
                    aux_relative_vector_loss += 5.0 * (angle_diff_pred - angle_diff_gt).abs().mean()
                
                if 'second_order' in clip_align_image_relative_vector:
                    n_sample = 30
                    rand_idx = torch.randperm(outputs_pairwise_diff.shape[0], device=outputs_pairwise_diff.device)[:n_sample]
                    rand_valid = torch.zeros_like(outputs_pairwise_diff[:,0,0]).bool()
                    rand_valid[rand_idx] = True
                    targets_chosen = targets[clip_correct_bool][rand_valid] # [n_sample]
                    outputs_norm_chosen = outputs_norm[clip_correct_bool][rand_valid] # [n_sample, d]
                    outputs_clip_norm_chosen = outputs_clip_norm[clip_correct_bool][rand_valid] # [n_sample, d]
                    cur_train_text_features_norm_chosen = cur_train_text_features_norm[torch.arange(n_sample).to(cuda_device), targets[clip_correct_bool][rand_valid]] # [n_sample, d]
                    if aux_clip_caption is not None:
                        aux_clip_caption_norm_chosen = aux_clip_caption_norm[rand_valid] # [n_sample, d]
                    else:
                        aux_clip_caption_norm_chosen = None
                    
                    rand_valid = rand_valid[:, None] * rand_valid[None, :]
                    outputs_pairwise_diff_chosen = outputs_pairwise_diff[rand_valid] # [n_sample * n_sample, d]
                    outputs_clip_pairwise_diff_chosen = outputs_clip_pairwise_diff[rand_valid] # [n_sample * n_sample, d]
                    
                    
                    rel_diff = (outputs_pairwise_diff_chosen - outputs_clip_pairwise_diff_chosen).norm(dim=-1) # [n_sample * n_sample] (ri-rj) - (ci-cj)
                    j_idx = torch.arange(n_sample, device=outputs_pairwise_diff.device).tile((n_sample,)) # [n_sample * n_sample]
                    i_idx = torch.arange(n_sample, device=outputs_pairwise_diff.device).repeat_interleave(n_sample)
                    targets_chosen_expand = targets_chosen.repeat_interleave(n_sample) # [n_sample * n_sample]
                    ij_idx = torch.arange(n_sample * n_sample, device=outputs_pairwise_diff.device)
                    # rel_diff = rel_diff + (outputs_clip_norm_chosen[i_idx] - cur_train_text_features_norm_chosen[i_idx]).norm(dim=-1)  
                    rel_diff_reshape = rel_diff.view(n_sample, n_sample) # [n_sample, n_sample]
                    rel_diff_for_loss = rel_diff_reshape.transpose(0, 1)[j_idx] # [n_sample * n_sample, n_sample]
                    if aux_clip_caption_norm_chosen is not None:
                        ck_m_ti = (
                            aux_clip_caption_norm_chosen[i_idx][:, None, :] 
                            - outputs_clip_norm_chosen[None, :, :]
                        ).norm(dim=-1)
                    else:
                        ck_m_ti = (
                            cur_train_text_features_norm_chosen[i_idx][:, None, :] 
                            - outputs_clip_norm_chosen[None, :, :]
                        ).norm(dim=-1)
                    rel_diff_for_loss = rel_diff_for_loss + ck_m_ti
                    rel_diff_for_loss = -rel_diff_for_loss # negative mse
                    if aux_clip_caption_norm_chosen is None:
                        mask = (targets_chosen_expand[:, None] == targets_chosen[None, :])
                        mask[ij_idx, i_idx] = False
                        rel_diff_for_loss[mask] = -1e7
                    aux_relative_vector_loss += 1.0 * F.cross_entropy(rel_diff_for_loss, i_idx, reduction='mean')
                    # rel_diff_for_loss = torch.clamp(rel_diff_for_loss - rel_diff_for_loss[ij_idx, i_idx][:, None] + 0.0, min=0.0)
                    # aux_relative_vector_loss += 1.0 * rel_diff_for_loss.sum(dim=-1).mean()
                    
                    # rel_diff = (
                    #     (outputs_pairwise_diff_chosen.view(n_sample, n_sample, -1).repeat_interleave(n_sample, dim=0)) 
                    #     - outputs_clip_pairwise_diff_chosen.view(n_sample, n_sample, -1).repeat_interleave(n_sample, dim=0)
                    # ).norm(dim=-1) # [n_sample * n_sample, n_sample]
                    # j_idx = torch.arange(n_sample, device=outputs_pairwise_diff.device).tile((n_sample,)) # [n_sample * n_sample]
                    # i_idx = torch.arange(n_sample, device=outputs_pairwise_diff.device).repeat_interleave(n_sample)
                    # ij_idx = torch.arange(n_sample * n_sample, device=outputs_pairwise_diff.device)
                    # ri_m_ti = (outputs_norm_chosen - cur_train_text_features_norm_chosen).norm(dim=-1) # [n_sample]
                    # ri_m_ti = ri_m_ti.repeat_interleave(n_sample) # [n_sample * n_sample]
                    # rel_diff = rel_diff + ri_m_ti[:, None]
                    # rel_diff[ij_idx, j_idx] = rel_diff[ij_idx, j_idx] - ri_m_ti
                    # rel_diff[ij_idx, j_idx] = rel_diff[ij_idx, j_idx] + (outputs_norm_chosen[j_idx] - cur_train_text_features_norm_chosen[i_idx]).norm(dim=-1)
                    # aux_relative_vector_loss += F.cross_entropy(-rel_diff / cur_temp, j_idx)
                    # aux_relative_vector_loss += torch.clamp(rel_diff[ij_idx, j_idx][:, None] - rel_diff, min=0.0).sum(dim=-1).mean()
                    
                    # print("i 11 j 23 k 15", rel_diff_for_loss[11 * n_sample + 23, 15], 
                    #       ((outputs_pairwise_diff_chosen[15 * n_sample + 23] - outputs_clip_pairwise_diff_chosen[15 * n_sample + 23]).norm(dim=-1) 
                    #        + (outputs_clip_norm_chosen[15] - cur_train_text_features_norm_chosen[11]).norm(dim=-1)
                    #       ) ,
                    # )
                    
                    # print("i 11 j 23 k 23", rel_diff_for_loss[11 * n_sample + 23, 23], 
                    #       ((outputs_pairwise_diff_chosen[23 * n_sample + 23] - outputs_clip_pairwise_diff_chosen[23 * n_sample + 23]).norm(dim=-1) 
                    #        + (outputs_clip_norm_chosen[23] - cur_train_text_features_norm_chosen[11]).norm(dim=-1)
                    #       ) ,
                    # )
                
                    # rel_diff_gt = (outputs_pairwise_diff_chosen - outputs_clip_pairwise_diff_chosen).norm(dim=-1) # [n_sample * n_sample]
                    # rel_diff_neg = ((outputs_pairwise_diff_chosen.view(n_sample, n_sample, -1).repeat_interleave(n_sample, dim=0))
                    #                  - outputs_clip_pairwise_diff_chosen[:, None, :]).norm(dim=-1) # [n_sample * n_sample, n_sample]
                    # aux_relative_vector_loss += torch.clamp(rel_diff_gt[:, None] - rel_diff_neg, min=0.0).sum(dim=-1).mean()
                
                # aux_relative_vector_loss = 5.0 * ((outputs_pairwise_diff - outputs_clip_pairwise_diff) ** 2).sum(dim=-1).mean()
        else:
            classify_outputs = outputs
            loss = criterion(classify_outputs, targets)
        # measure accuracy and record loss
        if loss is not None:
            losses.update(loss.item(), inputs.size(0))
        if aux_mse_loss is not None:
            aux_mse_losses.update(aux_mse_loss.item(), inputs.size(0))
        if aux_contrastive_loss is not None:
            aux_contrastive_losses.update(aux_contrastive_loss.item(), inputs.size(0))
        if aux_contrastive_prototype_loss is not None:
            aux_contrastive_prototype_losses.update(aux_contrastive_prototype_loss.item(), inputs.size(0))
        if aux_relative_vector_loss is not None:
            aux_relative_vector_losses.update(aux_relative_vector_loss.item(), inputs.size(0))
            
        if classify_outputs.ndim == 3:
            # [B, C, n_experts]
            classify_outputs_amax = classify_outputs.argmax(dim=1)
            tmp = torch.zeros_like(classify_outputs[:, :, 0])
            ones = torch.ones_like(classify_outputs_amax).float()
            tmp.scatter_(dim=1, index=classify_outputs_amax, src=ones, reduce='add')
            classify_outputs = tmp # [B, C]
            
        if avg_accuracy_per_class is None:
            avg_accuracy_per_class = [[0.0, 0.0] for _ in range(classify_outputs.shape[1])]
        for i in range(classify_outputs.shape[1]):
            outputs_this_class = classify_outputs[targets == i]
            if outputs_this_class.shape[0] > 0:
                avg_accuracy_per_class[i][0] += (outputs_this_class.argmax(dim=1) == i).sum().item()
                avg_accuracy_per_class[i][1] += outputs_this_class.shape[0]
        prec1 = accuracy(classify_outputs.data, targets.data, topk=(1,))[0]
        top1.update(prec1.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_for_bkward = 0.0
        if loss is not None:
            loss_for_bkward += loss
        if aux_mse_loss is not None:
            loss_for_bkward += aux_mse_loss
        if aux_contrastive_loss is not None:
            loss_for_bkward += aux_contrastive_loss
        if aux_contrastive_prototype_loss is not None:
            loss_for_bkward += aux_contrastive_prototype_loss
        if aux_relative_vector_loss is not None:
            loss_for_bkward += aux_relative_vector_loss
        total_losses.update(loss_for_bkward.item(), inputs.size(0))
        loss_for_bkward.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Tot Loss: {total_loss:.4f} | Loss: {loss:.4f} | Aux MSE Loss: {aux_mse_loss:.4f} | Aux contrastive Loss: {aux_contrastive_loss:.4f} | Aux contrastive Prototype Loss: {aux_contrastive_prototype_loss:.4f} | Aux relative vector Loss: {aux_relative_vector_loss:.4f} | Temperature: {temp:.4f} | top1 (non-cls-avg): {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    total_loss=total_losses.avg,
                    loss=losses.avg if loss is not None else 0.0,
                    aux_mse_loss=aux_mse_losses.avg if aux_mse_loss is not None else 0.0,
                    aux_contrastive_loss=aux_contrastive_losses.avg if aux_contrastive_loss is not None else 0.0,
                    aux_contrastive_prototype_loss=aux_contrastive_prototype_losses.avg if aux_contrastive_prototype_loss is not None else 0.0,
                    aux_relative_vector_loss=aux_relative_vector_losses.avg if aux_relative_vector_loss is not None else 0.0,
                    temp=args.temperature if not clip_align_image_contrastive_adaptive_temperature else cur_temp,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    
    avg_accuracy_per_class = [100.0 * x[0] / x[1] for x in avg_accuracy_per_class if x[1] > 0]
    print("Average accuracy per class: {}".format(avg_accuracy_per_class))
    mean_acc = np.mean(avg_accuracy_per_class)
    print("Mean accuracy per class {}".format(mean_acc))
    
    return (total_losses.avg, mean_acc)

def test(val_loader, model, criterion, epoch, use_cuda, 
         val_transform, val_text_features=None, clip_model=None, clip_preprocess=None, clip_device=None,
         chatgpt_text_features=None,
         chatgpt_ensemble=False,
         target_remap=None, 
         few_shot_features=None, support_set_idx=None, prompt_learner=None, text_encoder=None, prompt_mode='test'):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    avg_accuracy_per_class = None
            
    tot_idx = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if few_shot_features is not None:
            assert support_set_idx is not None
            # remove few-shot examples from the test set
            keep_bool = ~torch.isin((tot_idx + torch.arange(targets.shape[0])), support_set_idx)
            inputs = inputs[keep_bool]
            targets = targets[keep_bool] 
        tot_idx += targets.shape[0]
        
        inputs = val_transform(inputs)

        if use_cuda:
            inputs, targets = inputs.to(cuda_device), targets.to(cuda_device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            if hasattr(model, 'global_bias'):
                outputs = outputs + model.global_bias[None, :]
            if hasattr(model, 'global_bias_custom'):
                outputs = outputs + model.global_bias_custom[None, :]
            if chatgpt_text_features is not None:
                if not chatgpt_ensemble:
                    cur_val_text_features = chatgpt_text_features[:, -1, :]
                else:
                    cur_val_text_features = chatgpt_text_features
            else:
                cur_val_text_features = val_text_features
            if hasattr(model, 'txt_projection'):
                cur_val_text_features = model.txt_projection(cur_val_text_features)
            else:
                cur_val_text_features = val_text_features
            
        if clip_model is not None:
            outputs_norm = nn.functional.normalize(outputs, dim=1)
            ## input: normalized clip image features, output: normalized clip text features
            if prompt_learner is not None and text_encoder is not None:
                with torch.no_grad():
                    if prompt_mode == 'train':
                        tokenized_prompts = prompt_learner.train_tokenized_prompts
                    elif prompt_mode == 'test':
                        tokenized_prompts = prompt_learner.val_tokenized_prompts
                    elif prompt_mode == 'val_on_train':
                        tokenized_prompts = prompt_learner.val_on_train_tokenized_prompts
                    else: 
                        raise NotImplementedError()
                    prompts = prompt_learner(outputs_norm.to(cuda_device), mode=prompt_mode)
                    cur_val_text_features = []
                    for pts_i in prompts: # (n_cls, n_tkn, ctx_dim)
                        minib = 128
                        tokenized_idx = 0
                        cur_val_text_feature = []
                        while tokenized_idx < len(tokenized_prompts):
                            cur_val_text_feature.append(
                                text_encoder(pts_i[tokenized_idx:tokenized_idx+minib], tokenized_prompts[tokenized_idx:tokenized_idx+minib])
                            )
                            tokenized_idx += minib
                        cur_val_text_feature = torch.cat(cur_val_text_feature, dim=0)
                        cur_val_text_features.append(cur_val_text_feature[None, ...])
                    cur_val_text_features = torch.cat(cur_val_text_features, dim=0) # [B, n_cls, d]
                    cur_val_text_features_norm = nn.functional.normalize(cur_val_text_features, dim=-1)
                    classify_outputs = torch.einsum('ni,nci->nc', outputs_norm, cur_val_text_features_norm)
            else:
                cur_val_text_features_norm = nn.functional.normalize(cur_val_text_features, dim=-1)
                if cur_val_text_features_norm.ndim == 3:
                    classify_outputs = torch.einsum('ni,mci->nmc', outputs_norm, cur_val_text_features_norm)
                else:
                    classify_outputs = torch.einsum('ni,ci->nc', outputs_norm, cur_val_text_features_norm)
            if few_shot_features is None:
                classify_outputs = classify_outputs / args.temperature # temperature
            else:
                # few-shot following TIP-adapter
                beta = 5.5
                alpha = 1.0
                A = torch.exp(-beta * (1 - torch.einsum('ni,cmi->ncm', outputs_norm, few_shot_features))) # [B, n_class, few_shot_num]
                classify_outputs = classify_outputs + alpha * A.sum(dim=-1)
        else:
            classify_outputs = outputs
            if target_remap is not None:
                classify_outputs = classify_outputs[:, target_remap[0]:target_remap[1]]
            
        if classify_outputs.ndim == 2:
            loss = criterion(classify_outputs, targets)
        elif classify_outputs.ndim == 3:
            # classify_outputs shape [B, C, n_experts]
            loss = criterion(classify_outputs, targets[:, None].tile(1, classify_outputs.shape[-1]))
            classify_outputs_amax = classify_outputs.argmax(dim=1)
            tmp = torch.zeros_like(classify_outputs[:, :, 0])
            ones = torch.ones_like(classify_outputs_amax).float()
            tmp.scatter_(dim=1, index=classify_outputs_amax, src=ones, reduce='add')
            classify_outputs = tmp # [B, C]
            
        if avg_accuracy_per_class is None:
            avg_accuracy_per_class = [[0.0, 0.0] for _ in range(classify_outputs.shape[1])]
        for i in range(classify_outputs.shape[1]):
            classify_outputs_this_class = classify_outputs[targets == i]
            if classify_outputs_this_class.shape[0] > 0:
                avg_accuracy_per_class[i][0] += (classify_outputs_this_class.argmax(dim=1) == i).sum().item()
                avg_accuracy_per_class[i][1] += classify_outputs_this_class.shape[0]

        # measure accuracy and record loss
        prec1 = accuracy(classify_outputs.data, targets.data, topk=(1,))[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1 (non-cls-avg): {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    
    avg_accuracy_per_class = [100.0 * x[0] / x[1] for x in avg_accuracy_per_class if x[1] > 0]
    print("Average accuracy per class: {}".format(avg_accuracy_per_class))
    mean_acc = np.mean(avg_accuracy_per_class)
    print("Mean accuracy per class {}".format(mean_acc))
    return (losses.avg, mean_acc)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, scheduler=None):
    global state
    if epoch in args.schedule and scheduler is None:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
    if scheduler is not None and epoch > 0:
        scheduler.step()

if __name__ == '__main__':
    main()
