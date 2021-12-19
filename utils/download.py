from torchvision.datasets.utils import download_and_extract_archive, download_file_from_google_drive, \
    download_url, extract_archive
import os
import shutil
import torch
from utils.distributed import primary, synchronize


# These are the pre-trained GANgealing checkpoints we currently have available for download (and the SSL VGG network)
VALID_MODELS = {'bicycle', 'car', 'cat', 'cat_ssl_mix6', 'celeba', 'cub', 'dog', 'horse', 'tvmonitor',
                'simclr_vgg_phase150'}

# These are the default testing hyperparameters which we used for all models. They will be automatically
# loaded into the argparser whenever you use one of our pre-trained models. If you want to change any of these,
# add "--override" to your command. E.g., "--iters 7 --override".
# Note that we use the same padding_mode at test time as we did for training. We use iters=3 for the especially hard
# datasets (LSUN) and 1 otherwise.
PRETRAINED_TEST_HYPERPARAMS = \
    {
        'bicycle':       {'padding_mode': 'reflection', 'iters': 3},
        'car':           {'padding_mode': 'reflection', 'iters': 3, 'num_heads': 4},
        'cat':           {'padding_mode': 'border',    'iters': 3},
        'cat_ssl_mix6':  {'padding_mode': 'border',    'iters': 3},
        'celeba':        {'padding_mode': 'border',    'iters': 1},
        'cub':           {'padding_mode': 'border',    'iters': 1},
        'dog':           {'padding_mode': 'border',    'iters': 3},
        'horse':         {'padding_mode': 'reflection', 'iters': 3, 'num_heads': 4},
        'tvmonitor':     {'padding_mode': 'reflection', 'iters': 3},
    }


def find_model(model_name):
    if model_name in VALID_MODELS:
        using_pretrained_model = True
        return download_model(model_name), using_pretrained_model
    else:
        using_pretrained_model = False
        return torch.load(model_name, map_location=lambda storage, loc: storage), using_pretrained_model


def download_model(model_name, online_prefix='pretrained'):
    assert model_name in VALID_MODELS
    model_name = f'{model_name}.pt'  # add extension
    local_path = f'pretrained/{model_name}'
    if not os.path.isfile(local_path) and primary():  # download (only on primary process)
        web_path = f'http://efrosgans.eecs.berkeley.edu/gangealing/{online_prefix}/{model_name}'
        download_url(web_path, 'pretrained')
        local_path = f'pretrained/{model_name}'
    synchronize()  # Wait for the primary process to finish downloading the checkpoint
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


def download_spair(to_path):
    # Downloads and extracts the SPair-71K dataset
    spair_dir = f'{to_path}/SPair-71k'
    if not os.path.isdir(spair_dir):
        print(f'Downloading SPair-71k to {to_path}')
        spair_url = 'http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz'
        download_and_extract_archive(spair_url, to_path, remove_finished=True)
    else:
        print('Found pre-existing SPair-71K directory')
    return spair_dir


def download_lsun(to_path, category):
    lsun_dir = f'{to_path}/{category}'
    if not os.path.isdir(lsun_dir):
        lsun_url = f'http://dl.yf.io/lsun/objects/{category}.zip'
        download_and_extract_archive(lsun_url, to_path, remove_finished=True)
    else:
        print(f'Found pre-existing lsun {category} directory')
    return lsun_dir


def download_cub(to_path):
    # Downloads the CUB-200-2011 dataset
    cub_dir = f'{to_path}/CUB_200_2011'
    if not os.path.isdir(cub_dir):
        tgz_path = f'{cub_dir}.tgz'
        print(f'Downloading CUB_200_2011 to {to_path}')
        cub_file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
        download_file_from_google_drive(cub_file_id, to_path)
        shutil.move(f'{to_path}/{cub_file_id}', tgz_path)
        extract_archive(tgz_path, remove_finished=True)
    else:
        print('Found pre-existing CUB directory')
    return cub_dir


def download_cub_metadata(to_path):
    # Downloads some metadata so we can use image pre-processing consistent with ACSM for CUB
    acsm_val_mat_path = f'{to_path}/val_cub_cleaned.mat'
    if not os.path.isfile(acsm_val_mat_path):
        acsm_val_mat_url = f'http://efrosgans.eecs.berkeley.edu/gangealing/val_cub_cleaned.mat'
        print('Downloading metadata used to form ACSM\'s CUB validation set')
        download_url(acsm_val_mat_url, to_path)
    else:
        print('Found pre-existing CUB metadata')
    return acsm_val_mat_path


def download_video(video_name, online_prefix='video_1024'):
    valid_videos = {'elon', 'snowpuppy', 'cutecat'}
    assert video_name in valid_videos
    local_path = f'data/{video_name}'
    if not os.path.isdir(local_path) and primary():  # download (only on primary process)
        web_path = f'http://efrosgans.eecs.berkeley.edu/gangealing/{online_prefix}/{video_name}'
        os.makedirs(local_path)
        download_url(f'{web_path}/data.mdb', local_path)
        download_url(f'{web_path}/lock.mdb', local_path)
    return local_path


def download_lpips():
    local_path = f'pretrained/lpips_vgg_v0.1.pt'
    if not os.path.isfile(local_path) and primary():  # download (only on primary process)
        web_path = 'https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/vgg.pth'
        download_url(web_path, 'pretrained')
        shutil.move('pretrained/vgg.pth', local_path)
    synchronize()  # Wait for the primary process to finish downloading
