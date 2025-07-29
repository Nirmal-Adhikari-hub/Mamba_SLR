export OMP_NUM_THREADS=1

N_GPUS=2
MASTER_PORT=$((12000 + $RANDOM % 20000))


torchrun --nproc_per_node="$N_GPUS" --master_port="$MASTER_PORT" ../../main.py \
    --batch_size 2 \
    --epochs 100 \
    --update_freq 1 \
    --input_size 224 \
    --patch_size 16 \
    --d_model 512 \
    --headdim 64 \
    --depth 12 \
    --expand 2 \
    --head_drop_rate 0.1 \
    --drop_path 0.4 \
    --model_ema \
    --model_ema_decay 0.9999 \
    --model_ema_eval \
    --opt 'adamw' \
    --opt_betas 0.9 0.999 \
    --clip_grad 5.0 \
    --weight_decay 0.05 \
    --lr 1e-3 \
    --layer_decay 0.75 \
    --warmup_lr 1e-4 \
    --min_lr 1e-5 \
    --warmup_epochs 5 \
    --aa 'rand-m5-mstd0.25' \
    --temp_scale 0.0 \
    --model_key 'model_ema' \
    --prefix '/mnt/data/phoenix-2014/phoenix-2014-multisigner' \
    --gloss_dict_path '/home/chingiz/mambaslr/Mamba_SLR/data/phoenix2014/gloss_dict.npy' \
    --meta_dir_path '/home/chingiz/mambaslr/Mamba_SLR/data/phoenix2014' \
    --kp_path '/mnt/data/phoenix-2014/phoenix-2014-keypoints.pkl' \
    --output_dir '/home/chingiz/mambaslr/Mamba_SLR/slr/exp/pheonix-2014' \
    --log_dir '/home/chingiz/mambaslr/Mamba_SLR/slr/exp/pheonix-2014' \
    --resume '/home/chingiz/mambaslr/Mamba_SLR/slr/exp/pheonix-2014' \
    --auto_resume \
    --save_ckpt \
    --test_best \
    --num_workers 12 \
    --pin_mem \
    --enable_deepspeed \
    --bf16 \
    # --finetune '/home/kks/workspace/slr/image_backbone/exp/convnextStem-dim_512-depths_12-noz-cls/checkpoint-best-ema.pth' \