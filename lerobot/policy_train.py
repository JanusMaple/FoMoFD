"""
To train a policy, please run the `lerobot-train` command directly 
from the terminal with the appropriate config file and arguments. 

For example:

lerobot-train \
    --config_path configs/act_dinov2.yaml \
    --dataset.repo_id local/needle_pickup \
    --dataset.root ../fomofd/tasks/needle_pickup/data/processed_le_v21 \
    --output_dir ../fomofd/tasks/needle_pickup/model/act_dinov2 \
    --batch_size 64 \
    --steps 200000 \
    --save_freq 20000 \
    --log_freq 100 \
    --num_workers 8

lerobot-train \
    --config_path configs/act_dinov2_weak_fix.yaml \
    --dataset.repo_id local/needle_pickup \
    --dataset.root ../fomofd/tasks/needle_pickup/data/processed_le_v21 \
    --output_dir ../fomofd/tasks/needle_pickup/model/act_dinov2_weak_fix \
    --batch_size 64 \
    --steps 200000 \
    --save_freq 20000 \
    --log_freq 100 \
    --num_workers 8

lerobot-train \
    --config_path configs/act_dinov2_weak_wrist.yaml \
    --dataset.repo_id local/needle_pickup \
    --dataset.root ../fomofd/tasks/needle_pickup/data/processed_le_v21 \
    --output_dir ../fomofd/tasks/needle_pickup/model/act_dinov2_weak_wrist \
    --batch_size 64 \
    --steps 200000 \
    --save_freq 20000 \
    --log_freq 100 \
    --num_workers 8

lerobot-train \
    --config_path configs/act_dinov2_pretrained_tokens.yaml \
    --dataset.repo_id local/needle_pickup_pretrained_tokens \
    --dataset.root ../fomofd/tasks/needle_pickup/data/processed_le_v21_pretrained_tokens \
    --output_dir ../fomofd/tasks/needle_pickup/model/act_dinov2_pretrained_tokens \
    --batch_size 64 \
    --steps 100000 \
    --save_freq 2000 \
    --log_freq 100 \
    --num_workers 8

lerobot-train \
    --config_path configs/diffusion.yaml \
    --dataset.repo_id local/needle_pickup \
    --dataset.root ../fomofd/tasks/needle_pickup/data/processed_le_v21 \
    --output_dir ../fomofd/tasks/needle_pickup/model/diffusion \
    --batch_size 32 \
    --steps 20000 \
    --save_freq 2000 \
    --log_freq 50

lerobot-train \
    --config_path configs/vqbet_mv.yaml \
    --dataset.repo_id local/needle_pickup \
    --dataset.root ../fomofd/tasks/needle_pickup/data/processed_le_v21 \
    --output_dir ../fomofd/tasks/needle_pickup/model/vqbet_mv \
    --batch_size 32 \
    --steps 60000 \
    --save_freq 2000 \
    --log_freq 50
"""
