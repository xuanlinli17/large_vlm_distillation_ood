python robotics.py -a efficientnet_b0 \
    --train pickclutter_random_train_episodes_rgbd_512.h5 \
    --val pickclutter_random_val_episodes_rgbd_512.h5 \
    --val-on-train pickclutter_random_val_on_train_episodes_rgbd_512.h5 \
    --epochs 90 --lr 0.001  --gamma 0.3  --schedule 30 60  \
    -c checkpoints/effnet_chatgpt_classification_coop_contrastive_local_agg  \
    --gpu-id=0 --clip-gpu-id=0     --train-batch 60 --test-batch 40   \
    --clip-align-image-classification=1  --use-clip   --prompt-learner  \
    --clip-align-image-contrastive  --local-aggregation    \
    -chatgpt-raw-text-file   chatgpt.txt

python robotics.py -a efficientnet_b0 \
    --train pickclutter_random_train_episodes_rgbd_512.h5 \
    --val pickclutter_random_val_episodes_rgbd_512.h5 \
    --val-on-train pickclutter_random_val_on_train_episodes_rgbd_512.h5 \
    --epochs 20 --lr 0.0005  --onecyclelr  --gamma 0.3  --schedule 30 60  \
    -c checkpoints/effnet_chatgpt_classification_coop_contrastive_local_agg/fewshot50_coop_contrastive  \
    --gpu-id=0 --clip-gpu-id=0     --train-batch 46 --test-batch 40   \
    --clip-align-image-classification=1  --use-clip   --prompt-learner \
    --clip-align-image-contrastive --local-aggregation --few-shot-num 50 --few-shot-method finetune   \
    --resume checkpoints/effnet_chatgpt_classification_coop_contrastive_local_agg/checkpoint.pth.tar  \
    --chatgpt-raw-text-file   chatgpt.txt

# Need to set --train-batch to a multiple of 2

# Baselines

python robotics.py -a efficientnet_b0 \
    --train pickclutter_random_train_episodes_rgbd_512.h5 \
    --val pickclutter_random_val_episodes_rgbd_512.h5  \
    --val-on-train pickclutter_random_val_on_train_episodes_rgbd_512.h5   \
    --epochs 90 --lr 0.001  --gamma 0.3  --schedule 30 60  \
    -c checkpoints/effnet_classification_coop_localagg  \
    --gpu-id=0 --clip-gpu-id=0     --train-batch 60 --test-batch 40  \
    --clip-align-image-classification=1  --prompt-learner  --use-clip  --local-aggregation


python robotics.py -a efficientnet_b0 \
    --train pickclutter_random_train_episodes_rgbd_512.h5 \
    --val pickclutter_random_val_episodes_rgbd_512.h5 \
    --val-on-train pickclutter_random_val_on_train_episodes_rgbd_512.h5 \
    --epochs 20 --lr 0.0005  --onecyclelr --gamma 0.3  --schedule 30 60  \
    -c checkpoints/effnet_classification_coop_localagg/fewshot50_coop  \
    --gpu-id=0 --clip-gpu-id=0     --train-batch 46 --test-batch 40 \
    --resume checkpoints/effnet_classification_coop_localagg/checkpoint.pth.tar \
    --clip-align-image-classification=1  --use-clip  --local-aggregation  \
    --prompt-learner --few-shot-num 50 --few-shot-method finetune