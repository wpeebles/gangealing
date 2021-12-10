python -m torch.distributed.launch --nproc_per_node=8 --master_port=6085 train.py \
--ckpt lsun_cars_stylegan2.pt --padding_mode reflection \
--vis_every 5000 --ckpt_every 50000 --iter 1500000 --loss_fn lpips --exp-name lsun_cars \
--num_heads 4 --flips --ndirs 5 --inject 6 --sample_from_full_res
