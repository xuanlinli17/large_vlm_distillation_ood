# Distilling Large Vision-Language Model With Out-of-Distribution Generalizability

[paper]()

Large vision-language models have achieved outstanding performance, but their size and computational requirements make their deployment on resource-constrained devices and time-sensitive tasks impractical. In this paper, we investigate the distillation of visual representations in large teacher vision-language models into lightweight student models using a small- or mid-scale dataset, aiming to maintain the performance of teacher models. Notably, this study focuses on open-vocabulary out-of-distribution (OOD) generalization, a challenging problem that has been overlooked in previous model distillation literature. We propose two principles from vision and language modality perspectives to enhance student's OOD generalization: **(1)** by better imitating teacher's visual representation space, and carefully promoting better coherence in vision-language alignment with the teacher; **(2)** by enriching the teacher's language representations with informative and finegrained semantic attributes to effectively distinguish between different labels. We propose several metrics and conduct extensive experiments to investigate their techniques. The results demonstrate significant improvements in zero-shot and few-shot student performance on open-vocabulary out-of-distribution classification.

- [Distilling Large Vision-Language Model With Out-of-Distribution Generalizability](#distilling-large-vision-language-model-with-out-of-distribution-generalizability)
    - [Setup](#setup)
    - [Preparing datasets](#preparing-datasets)
    - [Running main experiments](#running-main-experiments)
    - [Citations](#citations)
    - [License](#license)


### Setup

First create an anaconda environment:

`conda create -n large_vlm_distillation_ood python=3.8`

Then clone this repository and install it:

```
git clone https://github.com/xuanlinli17/large_vlm_distillation_ood
pip install -e .
```

### Preparing datasets



### Running main experiments

To train a student, run the following command:

```
python main_experiments.py -d {path_to_dataset} \
    -a {student arch: e.g., resnet18 or vit_b_32} \
    --repeat-epochs {repeat_epochs} \
    {schedule_commands} \
    -c {save_path} \
    {batch_args} \
    {clip_model_args} \
    {loss_option_args}
```

where, for example,
- For ResNet students, `repeat_epochs` equals 5 for small datasets (Flowers, Cars, Birds), and 1 for larger datasets (Food, SUN, Tiered-ImageNet)
- For ViT students, `repeat_epochs` equals 5 for small datasets, and 3 for larger datasets
- `schedule_commands` equals `--epochs 90 --lr 0.05 --schedule 30 60 --gamma 0.1` for ResNet students and `--epochs 90 --repeat-epochs 3 --lr 0.0001 --onecyclelr --use-adam` for ViT students.
- `batch_args` equals `--train-batch 128 --test-batch 64`.
- `clip_model_args` equals `--use-clip --clip-model ViT-L/14`.
- `loss_option_args` can be any number of the following options:
  - adding `--clip-align-image-classification=1` enables `L-cls` (vision-language alignment loss for classification). Setting `--clip-align-image-classification=0` turns it off.
  - adding `--clip-align-image-mse` enables `L-mse` which naively matches teacher visual features
  - adding `--clip-align-image-contrastive` enables `L-cst`
  - adding `--clip-align-proximal-text-num=256` enables `L-vlalign` with `k=256`. Further adding `--clip-filter-out-wrong-alignment` enables the filtering out of images misaligned with language labels in `L-vlalign`.
  - adding `--chatgpt-raw-text-file {path to chatgpt.txt}` enables chatgpt-generated label descriptions
  - adding `--clip-align-image-aux-caption` enables auxiliary captions
  - adding `--prompt-learner` enables prompt learning through CoOp

To few-shot finetune a student, the commands are similar to the above, except that:
- Replace `schedule_commands` as `--epochs 20 --lr 0.001 --onecyclelr --gamma 0.1` for ResNet students and `--epochs 20 --repeat-epochs 3 --lr 0.0002 --onecyclelr --use-adam` for ViT students.
- Replace `-c` as `{previous_save_path}/fewshot5`
- Add these additional args: `--few-shot-num 5 --few-shot-method finetune`
- Add checkpoint to the args: `--resume {previous_save_path}/checkpoint.pth.tar`

A few more example commands are shown in `example_scripts.sh`.


### Citations

Please cite our paper if you find our idea helpful. Thanks a lot!

```
TBD
```

### License

This project is licensed under the MIT license.