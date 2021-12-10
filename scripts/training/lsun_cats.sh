python -m torch.distributed.launch --nproc_per_node=8 --master_port=6085 train.py \
--ckpt lsun_cats_stylegan2.pt --padding_mode border --vis_every 5000 --ckpt_every 50000 \
--iter 1500000 --loss_fn lpips --exp-name lsun_cats