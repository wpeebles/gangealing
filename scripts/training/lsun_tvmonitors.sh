torchrun --nproc_per_node=8 train.py \
--ckpt tvmonitor --load_G_only --num_fp16_res 4 --padding_mode reflection --tv_weight 2500 \
--vis_every 5000 --ckpt_every 50000 --iter 1500000 --loss_fn lpips --exp-name lsun_tvs
