## GAN-Supervised Dense Visual Alignment &mdash; Official PyTorch Implementation

[Paper](https://arxiv.org/abs/2112.05143) | [Project Page](https://www.wpeebles.com/gangealing) | [Video](https://youtu.be/Qa1ASS_NuzE)

![Teaser image](images/elon.gif) ![Teaser image](images/catpet2teaser.gif)

This repo contains training, evaluation, and visualization code for the GANgealing algorithm from our GAN-Supervised Dense Visual Alignment paper.

[**GAN-Supervised Dense Visual Alignment**](https://www.wpeebles.com/gangealing)<br>
William Peebles, Jun-Yan Zhu, Richard Zhang, Antonio Torralba, Alexei Efros, Eli Shechtman<br>
UC Berkeley, Carnegie Mellon University, Adobe Research, MIT CSAIL<br>

GAN-Supervised Learning is a method for learning discriminative models and their GAN-generated training data jointly end-to-end. We apply our framework to the dense visual alignment problem. Inspired by the classic Congealing method, our GANgealing algorithm trains a Spatial Transformer to
warp random samples from a GAN trained on unaligned data to a common, jointly-learned target mode. The target mode is
updated to make the Spatial Transformer's job "as easy as possible." The Spatial Transformer is trained exclusively on GAN images and generalizes
to real images at test time automatically.

[![Watch the video](images/method.png)](https://www.wpeebles.com/images/gangealing_visuals/gangealing.mp4)

This repository contains:

* 🎱 Pre-trained GANgealing models for eight datasets, including both the Spatial Transformers and generators
* 💥 Training code which fully supports Distributed Data Parallel
* 🎥 Scripts for running our Mixed Reality application with pre-trained Spatial Transformers
* ⚡ A lightning-fast CUDA implementation of splatting to generate high-quality warping visualizations
* 🎆 Several additional evaluation and visualization scripts to reproduce results from our paper and website

This codebase should be mostly ready to go, but we may make a few tweaks over December 2021 to smooth out any remaining wrinkles.

## Setup

First, download the repo:

```bash
git clone git@github.com:wpeebles/gangealing.git
cd gangealing
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment:

```bash
conda env create -f environment.yml
conda activate gg
```

If you use your environment, we recommend using the most current version of PyTorch.

## Running Pre-Trained Models

The [`applications`](applications) directory contains several files for evaluating and visualizing pre-trained GANgealing models.

We provide several pre-trained GANgealing models: `bicycle`, `cat`, `celeba`, `cub`, `dog` and `tvmonitor`. We also have pre-trained checkpoints
for our `car` and `horse` clustering models. Calling any of the files in `applications` with the `--ckpt` argument will automatically download and cache
the weights. As described in our paper, we highly recommend using `--iters 3`, which controls the number of times the similarity Spatial Transformer is recursively evaluated, for all LSUN models to get the most accurate results (and `--iters 1` for In-The-Wild CelebA and CUB). Finally, the `--output_resolution` argument controls the size of congealed images output by the Spatial Transformer. For the highest quality results, we recommend setting this equal to `--real_size` (default value is 128).

### Preparing Real Data

We use LMDBs for storing data. You can use [`prepare_data.py`](prepare_data.py) to pre-process input datasets. Note that setting-up real data is not
required for training.


**LSUN:** Download and unzip the relevant category from [here](http://dl.yf.io/lsun/objects/) (e.g., `cat`). You can pre-process the data with the following command:

```python
python prepare_data.py --input_is_lmdb --path path_to_downloaded_folder --out data/lsun_cats --pad center --size 512
```

**Image Folders:** For any dataset where you have all images in a single folder, you can pre-process them with:

```python
python prepare_data.py --path folder_of_images --out data/my_new_dataset --pad [center/border/zero] --size S
```

where `S` is the square resolution of the resized images.

**SPair-71K:** You can download and prepare SPair for PCK evaluation (e.g., for Cats) with:

```python
python prepare_data.py --spair_category cat --spair_split test --out data/spair_cats_test --size 256
```

**CUB:** We closely follow the pre-processing steps used by [ACSM](https://github.com/nileshkulkarni/acsm) for CUB PCK evaluation. You can download and prepare the CUB validation split with:

```python
python prepare_data.py --cub_acsm --out data/cub_val --size 256
```

### Congealing and Dense Correspondence Visualization

![Teaser image](images/cats_cube_small.gif)

`vis_correspondence.py` produces a video depicting real images being gradually aligned with our Spatial Transformer network.
It also can be used to visualize label/object propagation:

```python
python applications/vis_correspondence.py --ckpt cat --iters 3 --real_data_path data/lsun_cats --vis_in_stages --real_size 512 --output_resolution 512 --resolution 512 --label_path assets/masks/cat_mask.png --dset_indices 1922 2363 8558 7401 9750 7432 2105 53 1946
```

### Mixed Reality (Object Lenses)

![Teaser image](images/catpet2teaser.gif)

`mixed_reality.py` applies a pre-trained Spatial Transformer per-frame to an input video. We include several objects
and masks you can propagate in the `assets` folder.

The first step is to prepare the video dataset. If you have the video saved as an image folder (with filenames in order based on timestamp), you can run:

```python
python prepare_data.py --path folder_of_frames --out data/my_video_dataset --pad center --size 1024
```
This command will pre-process the images to square with center-cropping and resize them to 1024x1024 resolution (you can use any square resolution you like).
You can also instead specify `--pad border` to perform border padding instead of cropping.

If your video is saved in `mp4`, `mov`, etc. format, we provide a script that will convert it into frames via FFmpeg:

```bash
./process_video.sh path_to_video
```

This will save a folder of frames in the `data/video` folder, which you can then run `prepare_data.py` on as described above.

Now that the data are set up, we can run GANgealing on the video. For example, this will propagate a cartoon face via our LSUN Cats
model:

```python
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --master_port=6085 applications/mixed_reality.py --ckpt cat --iters 3 --objects --label_path assets/objects/cat/cat_cartoon.png --sigma 0.3 --opacity 1 --real_size 1024 --resolution 8192 --out video_materials_full/cats --real_data_path path_to_my_video --no_flip_inference
```

This will efficiently parallelize the evaluation of the video over `NUM_GPUS`. If you are propagating to a long video/ are running out of memory, you can add the `--save_frames` argument which should use significantly less memory (at the cost of speed). The `--objects` argument pulls propagated RGB values from the RGBA image `--label_path` points to. If you omit `--objects`, only the alpha channel of `--label_path` will be used, and a colorscale will be created (useful for visualizing tracking when propagating masks). For models that do not benefit much from flipping (e.g., LSUN Cats, TVs, and CelebA), we recommend using the `--no_flip_inference` argument to disable unnecessary flipping.

#### Creating New Object Lenses

To propagate your own custom object, you need to create a new RGBA image saved as a `png`. You can take the
pre-computed average congealed image for your model of interest (located in [`assets/averages`](assets/averages)) and load it
into an image editor like Photoshop. Then, overlay your object of interest on the template and export the object as an RGBA `png` image.
Pass your new object with the `--label_path` argument like above.

We recommend saving the object at a high resolution for the highest quality results (e.g., 4K resolution or higher if you are propagating to a 1K resolution video).

### PCK-Transfer Evaluation

Our repo includes a fast implementation of PCK-Transfer in `pck.py` that supports multi-GPU evaluation. First, make sure you've set up either SPair-71K or CUB as described earlier. You can evaluate PCK-Transfer as follows:

To evaluate SPair-71K (e.g., `cats` category):

```python
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --master_port=6085 applications/pck.py --ckpt cat --iters 3 --real_data_path data/spair_cats_test --real_size 256
```

To evaluate PCK on CUB:

```python
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --master_port=6085 applications/pck.py --ckpt cub --real_data_path data/cub_val --real_size 256 --num_pck_pairs 10000 --transfer_both_ways
```

You can also add the `--vis_transfer` argument to save a visualization of keypoint transfer.

Note that different methods compute PCK in slightly different ways depending on dataset. For CUB, the protocol used by past methods is to sample 10,000 random pairs from the validation set and evaluate bidirectional transfers. For SPair, fixed pairs are always used and the transfers are one-way. Our implementation of PCK supports both of these protocols to ensure accurate comparisons against baselines.

### Learned Pre-Processing of Datasets

Finally, we also include a script that applies a pre-trained Spatial Transformer to align and filter a dataset (e.g., for downstream GAN training).

To do this, you will need two versions of your dataset: (1) a pre-processed version (via `prepare_data.py` as described above) which will be used to quickly compute flow smoothness scores, and (2) a raw, unprocessed version of the dataset stored in LMDB format. We'll explain how to create this second unprocessed copy below.

The first recommended step is to compute _flow smoothness scores_ for each image in the dataset. As described in our paper, these scores
do a good job at identifying (1) images the Spatial Transformer fails on and (2) images that are impossible to align to the learned target mode. The scores can be computed as follows:

```python
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --master_port=6085 applications/flow_scores.py --ckpt cat --iters 3 --real_data_path my_dataset --real_size S --no_flip_inference
```
, where `my_dataset` should be created with our `prepare_data.py` script as described above. This will cache a tensor of flow scores at `my_dataset/flow_scores.pt`.

Next is the alignment step. Create an LMDB of the raw, unprocessed images in your unaligned dataset using the `--pad none` argument:

```python
python prepare_data.py --path folder_of_frames --out data/new_lmdb_data --pad none --size 0
```
Finally, you can generate a new, aligned and filtered dataset:

```python
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --master_port=6085 applications/congeal_dataset.py --ckpt cat --iters 3 --real_data_path data/new_lmdb_data --out data/my_new_aligned_dataset --real_size 0 --flow_scores my_dataset/flow_scores.pt --fraction_retained 0.25 --output_resolution S
```
, where `S` is the desired output resolution of the dataset and the `--fraction_retained` argument controls the percentage of images that will be retained based on flow scores. There are some other arguments you can adjust---see documentation in `congeal_dataset.py` for details.


### Using Pre-Trained Clustering Models

The clustering models are usable in most places the unimodal models are (with a few current exceptions, such as `flow_scores.py` and `congeal_dataset.py`). To load the clustering models, add `--num_heads K` (for our pre-trained models, `K=4`). There are also several files that let you propagate from a chosen cluster with the `--cluster <cluster_index>` argument (e.g., `mixed_reality.py` and `vis_correspondence.py`). Please refer to the documentation in these files for details.

## Training

(We will add additional training scripts in the coming days!)

To train new GANgealing models, you will need pre-trained StyleGAN2(-ADA) generator weights from the [rosinality repo](https://github.com/rosinality/stylegan2-pytorch). We also include generator checkpoints in all of our pre-trained GANgealing weights. Please refer to the `scripts` folder for examples of training commands, and see `train.py` for details.

When training a clustering model `(--num_heads > 1)`, you will need to train a cluster classifier network to use the model on real images. This is done with `train_cluster_classifier.py`; see example commands in `scripts`.

Note that for the majority of experiments in our paper, we trained using 8 GPUs and a per-GPU batch size of 5.

## Citation

If our code or models aided your research, please cite our [paper](https://arxiv.org/abs/2112.05143):
```
@article{peebles2021gansupervised,
title={GAN-Supervised Dense Visual Alignment},
author={William Peebles and Jun-Yan Zhu and Richard Zhang and Antonio Torralba and Alexei Efros and Eli Shechtman},
year={2021},
journal={arXiv preprint arXiv:2112.05143},
}
```

## Acknowledgments

We thank Tim Brooks for his antialiased sampling code and helpful discussions. We thank Tete Xiao, Ilija Radosavovic, Taesung Park, Assaf Shocher, Phillip Isola, Angjoo Kanazawa, Shubham Goel, Allan Jabri, Shubham Tulsiani, and Dave Epstein for helpful discussions. This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE 2146752. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.  Additional funding is provided by Adobe and Berkeley Deep Drive.

This repository is built on top of rosinality's excellent [PyTorch re-implementation of StyleGAN2](https://github.com/rosinality/stylegan2-pytorch).
