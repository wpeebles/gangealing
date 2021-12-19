torchrun --nproc_per_node=8 train.py \
--ckpt dog --load_G_only --padding_mode border --tv_weight 2500 \
--vis_every 5000 --ckpt_every 50000 --iter 1500000 --loss_fn lpips --exp-name lsun_dogs
