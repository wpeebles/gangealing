python -m torch.distributed.launch --nproc_per_node=8 --master_port=6085 train.py \
--ckpt lsun_bicycles_stylegan2.pt --num_fp16_res 4 --padding_mode reflection \
--vis_every 5000 --ckpt_every 50000 --iter 1500000 --loss_fn lpips --exp-name lsun_bicycles
