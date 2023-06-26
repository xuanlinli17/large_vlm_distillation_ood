'''
Example:
python evaluate_clip.py --data /home/xuanlin/102flowers/splited55/  --batch-size 96 --label-path /home/xuanlin/102flowers/splited55/label2text.txt  \
    --chatgpt-raw-text-file /home/xuanlin/102flowers/splited55/chatgpt.txt  --gpu-id 1
'''

from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch, numpy as np
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

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# Parse arguments
parser = argparse.ArgumentParser(description='Zero-shot CLIP evaluation')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch-size', default=96, type=int, metavar='N')
# Openset-specific
parser.add_argument('--label-path', type=str, default=None, help='path to label text file')
parser.add_argument('--chatgpt-raw-text-file', type=str, default=None)
parser.add_argument('--clip-repo', type=str, default='clip', choices=['clip', 'open_clip'])
parser.add_argument('--clip-model', type=str, default='ViT-L/14')
parser.add_argument('--clip-dataset', type=str, default='openai', choices=['openai', 'laion400m_e31', 'laion400m_e32', 'laion2b_s32b_b82k'])
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

if use_cuda:
    cuda_device = f"cuda:{args.gpu_id}"

def main():
    
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    val_on_train_dir = os.path.join(args.data, 'val_on_train')
    
    train_dataset = CLIPImageDataset(traindir, 
                                    transforms.Compose([
                                        transforms.Resize([256, 256]),
                                        transforms.ToTensor()
                                    ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_dataset = CLIPImageDataset(valdir, 
                                    transforms.Compose([
                                        transforms.Resize([256, 256]),
                                        transforms.ToTensor()
                                    ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, shuffle=False,
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
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    
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
            line = line.strip().split(' ') # [dir_name, id, natural_language_label]
            if len(line) > 0:
                line[2] = line[2].replace('_', ' ')
                label2text[line[0]] = line[2]
                if args.chatgpt_raw_text_file is not None:
                    chatgpt_label2text[line[0]] = chatgpt_lines[idx]
                idx += 1
    if args.chatgpt_raw_text_file is not None:
        assert len(list(label2text.keys())) == len(chatgpt_lines), f"{len(label2text.keys())} != {len(chatgpt_lines)}"
        
    print("train class_to_idx", train_dataset.class_to_idx)
    print("val class_to_idx", val_dataset.class_to_idx)
    
    if args.chatgpt_raw_text_file is not None:
        gen_text_fxn = lambda x: label2text[x] + " . " + chatgpt_label2text[x]
    else:
        gen_text_fxn = lambda x: label2text[x]
    train_text_labels = ["a photo of " + gen_text_fxn(x) for x in train_dataset.class_to_idx.keys()]
    val_text_labels = ["a photo of " + gen_text_fxn(x) for x in val_dataset.class_to_idx.keys()]
    
    print("train_text_labels", train_text_labels)
    print("val_text_labels", val_text_labels)
    
    if args.clip_repo == 'clip': 
        train_text_features = clip_model.encode_text(clip.tokenize(train_text_labels, truncate=True).to(clip_device)).float().detach()
        val_text_features = clip_model.encode_text(clip.tokenize(val_text_labels, truncate=True).to(clip_device)).float().detach()
    elif args.clip_repo == 'open_clip':
        tokenize = open_clip.tokenizer.tokenize
        train_text_features = clip_model.encode_text(tokenize(train_text_labels).to(clip_device)).float().detach()
        val_text_features = clip_model.encode_text(tokenize(val_text_labels).to(clip_device)).float().detach()
        
    criterion = nn.CrossEntropyLoss()


    print('\nEvaluation only')
    test_loss, test_acc = test(train_loader, criterion, train_text_features, clip_model, clip_preprocess, clip_device,)
    print(' Training set Loss:  %.8f, Training set Acc:  %.2f' % (test_loss, test_acc))
    if val_on_train_loader is not None:
        test_loss, test_acc = test(val_on_train_loader, criterion, train_text_features, clip_model, clip_preprocess, clip_device,)
        print(' In-distribution set Loss:  %.8f, In-distribution set Acc:  %.2f' % (test_loss, test_acc))
    test_loss, test_acc = test(val_loader, criterion, val_text_features, clip_model, clip_preprocess, clip_device,)
    print(' Test set Loss:  %.8f, Test set Acc:  %.2f' % (test_loss, test_acc))
    return
    

def test(val_loader, criterion, val_text_features, clip_model, clip_preprocess, clip_device):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    avg_accuracy_per_class = None
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = clip_preprocess(inputs)

        if use_cuda:
            inputs, targets = inputs.to(cuda_device), targets.to(cuda_device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = clip_model.encode_image(inputs).float()
        outputs_norm = nn.functional.normalize(outputs, dim=1)
        val_text_features_norm = nn.functional.normalize(val_text_features, dim=1)
        outputs = torch.einsum('ni,mi->nm', outputs_norm, val_text_features_norm)
        outputs = outputs / 0.01 # temperature
        # print(outputs.softmax(dim=1))
        loss = criterion(outputs, targets)
        if avg_accuracy_per_class is None:
            avg_accuracy_per_class = [[0.0, 0.0] for _ in range(outputs.shape[1])]
        for i in range(outputs.shape[1]):
            outputs_this_class = outputs[targets == i]
            if outputs_this_class.shape[0] > 0:
                avg_accuracy_per_class[i][0] += (outputs_this_class.argmax(dim=1) == i).sum().item()
                avg_accuracy_per_class[i][1] += outputs_this_class.shape[0]

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
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
    print("Num samples per class: ", [x[1] for x in avg_accuracy_per_class])
    avg_accuracy_per_class = [100.0 * x[0] / (x[1] + 1e-6) for x in avg_accuracy_per_class]
    print("Average accuracy per class: {}".format(avg_accuracy_per_class))
    mean_acc = np.mean(avg_accuracy_per_class)
    print("Mean accuracy per class (sum(accuracy) / n_classes): {}".format(mean_acc))
    return (losses.avg, mean_acc)



if __name__ == '__main__':
    main()
