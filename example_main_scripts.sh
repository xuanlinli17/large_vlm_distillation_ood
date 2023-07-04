python imagenet.py -a resnet18 --data food/splited --label-path food/splited/label2text.txt --epochs 90 \
    --lr 0.05 --schedule 30 60 --gamma 0.1 -c /home/some_save_path/ \
    --use-clip --train-batch 128 --test-batch 64 --clip-align-image-classification=1 \
    --clip-align-image-mse --clip-align-image-contrastive \
    --clip-align-proximal-text-num 256 --clip-filter-out-wrong-alignment \
    --chatgpt-raw-text-file food/splited/chatgpt.txt

python imagenet.py -a resnet18 --data food/splited --label-path food/splited/label2text.txt --epochs 20 \
    --lr 0.001 --onecyclelr --gamma 0.1 -c /home/some_save_path/fewshot5 \
    --train-batch 128 --test-batch 64 --use-clip --clip-align-image-classification=1 \
    --clip-align-image-mse --clip-align-image-contrastive --chatgpt-raw-text-file food/splited/chatgpt.txt \
    --clip-align-proximal-text-num 256 --clip-filter-out-wrong-alignment \
    --few-shot-num 5 --few-shot-method finetune \
    --resume /home/some_save_path/checkpoint.pth.tar

python imagenet.py -a vit_b_32 --data food/splited --label-path food/splited/label2text.txt \
    --epochs 90 --repeat-epochs 3 --lr 0.0001 --onecyclelr --use-adam \
    -c /home/some_save_path_2/ \
    --train-batch 128 --test-batch 64 --use-clip --clip-align-image-classification=1 \
    --clip-align-proximal-text-num 256 --clip-filter-out-wrong-alignment \
    --clip-align-image-contrastive --clip-align-image-mse

python imagenet.py -a vit_b_32 --data food/splited --label-path food/splited/label2text.txt \
    --epochs 20 --repeat-epochs 3 --lr 0.0002 --onecyclelr --use-adam \
    -c /home/some_save_path_2/fewshot5 \
    --train-batch 128 --test-batch 64 --few-shot-num 5 --few-shot-method finetune \
    --use-clip --clip-align-image-classification=1 --clip-align-image-contrastive --clip-align-image-mse \
    --clip-align-proximal-text-num 256 --clip-filter-out-wrong-alignment \
    --resume /home/some_save_path_2/checkpoint.pth.tar