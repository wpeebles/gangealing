python -m torch.distributed.launch --nproc_per_node=8 --master_port=6085 train_cluster_classifier.py \
--ckpt lsun_cars_gangealing_checkpoint.pt --padding_mode reflection \
--vis_every 5000 --ckpt_every 50000 --iter 55000 --period 50000 --loss_fn lpips --exp-name lsun_cars_cluster_classifier \
--num_heads 4 --flips --ndirs 5 --inject 6 --sample_from_full_res
