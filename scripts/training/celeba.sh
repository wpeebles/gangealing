# Note: if you're training with fewer than 8 gpus, you should increase the per-gpu batch size controlled by
# the --batch argument so total batch size is preserved. Default value of --batch is 5 (assumes 8 gpus for training
# for a total batch size of 40 across all gpus)
torchrun --nproc_per_node=8 train.py \
--ckpt celeba --load_G_only --padding_mode border --gen_size 128 --vis_every 5000 --ckpt_every 50000 \
--iter 1500000 --tv_weight 2500 --ndirs 512 --inject 6 --loss_fn lpips --exp-name in_the_wild_celeba
