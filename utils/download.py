from torchvision.datasets.utils import download_and_extract_archive, download_file_from_google_drive, \
    download_url, extract_archive
import os
import shutil
import torch

# These are the pre-trained GANgealing checkpoints we currently have available for download:
VALID_MODELS = {'bicycle', 'car', 'cat', 'cat_ssl_mix6', 'celeba', 'cub', 'dog', 'horse', 'tvmonitor'}


def find_model(model_name):
    if model_name in VALID_MODELS:
        return download_model(model_name)
    else:
        return torch.load(model_name, map_location=lambda storage, loc: storage)


def download_model(model_name):
    assert model_name in VALID_MODELS
    model_name = f'{model_name}.pt'  # add extension
    local_path = f'pretrained/{model_name}'
    if os.path.isfile(local_path):
        model = torch.load(local_path, map_location=lambda storage, loc: storage)
    else:
        web_path = f'http://efrosgans.eecs.berkeley.edu/gangealing/pretrained/{model_name}'
        download_url(web_path, 'pretrained')
        local_path = f'pretrained/{model_name}'
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


def download_cub(to_path):
    # Downloads the CUB-200-2011 dataset
    cub_dir = f'{to_path}/CUB_200_2011'
    if not os.path.isdir(cub_dir):
        zip_path = f'{cub_dir}.zip'
        print(f'Downloading CUB_200_2011 to {to_path}')
        cub_file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
        download_file_from_google_drive(cub_file_id, to_path)
        shutil.move(f'{to_path}/{cub_file_id}', zip_path)
        extract_archive(zip_path, remove_finished=True)
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
