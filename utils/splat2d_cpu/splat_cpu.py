'''/*
 * File   : splat_cpu.py
 * Author : Bill Peebles
 * Email  : peebles@berkeley.edu
 * Modify to python : Rebecca Li
 * Email  : xiaoli@adobe.com
 */
'''
import torch
import numpy as np

def GaussianPDF(  mu_1:float, mu_2, x_1, x_2, normalizer):
    '''
    Calculate the gausian normalized pixel value
    '''
    return torch.exp(normalizer * (torch.pow(x_1 - mu_1, 2.0) + torch.pow(x_2 - mu_2, 2.0)))

def SplatForward(
        bottom_coordinates,
        bottom_values,
        bottom_sigma: float,
         top_alpha_splats: float,
         top_output: float,
         num_points : int,
         channels  : int,
         height  : int,
         width  : int):

    '''
    Modify from  https://github.com/wpeebles/gangealing/blob/main/utils/splat2d_cuda/src/splat_gpu_impl.cu
    # self is a loop over batch and point in coordinates

    Args:
        bottom_coordinates: (n, num_points, 2), (x,y)-coordinates
        bottom_values: (n, num_points, channels)
        bottom_sigma: float
        top_alpha_splats: (n, height,width)
        top_output: (N, C, H, W)
    Output:
        top_alpha_splats: updated value.
        top_output: updated value. (N, C, H, W)  an element in the output

    # pw = index % width
    # ph = (index / width) % height
    # c = (index / width / height) % channels

    [TODO] use CPU multithread to speed up this function
    reference: https://docs.python.org/3/library/multiprocessing.html

    - Test the function:    

        .. code-block::

            unit_tests/test_splat.py
    '''
    for n in range( int(bottom_coordinates[0])):
        # n for batch id
        for i in range(num_points):
            # i for point id
            stdev = bottom_sigma[n]
            length = 2 * stdev
            x_coord = bottom_coordinates[n][i][0]
            y_coord = bottom_coordinates[n][i][1]
            normalizer = - torch.pow(2 * stdev * stdev, -1.0)

            # Ignore out-of-bounds points:
            if (x_coord >= 0 and x_coord < width) and (y_coord >= 0 and y_coord < height):

                # import pdb;pdb.set_trace()
                t = int(torch.fmax(torch.tensor(0), torch.floor(y_coord - length)))
                b = int(torch.fmin(torch.tensor(height - 1), torch.ceil(y_coord + length)))
                l = int(torch.fmax(torch.tensor(0), torch.floor(x_coord - length)))
                r = int(torch.fmin(torch.tensor(width - 1), torch.ceil(x_coord + length)))

                for lh in range ( t,b+1):
                    for lw  in range (l, r+1 ) :               
                        alpha = GaussianPDF(x_coord, y_coord, float(lw), float(lh), normalizer)
                        current_alpha_splat = top_alpha_splats[n][lh][lw]
                        top_alpha_splats[n][lh][lw] = current_alpha_splat + alpha
                        for c in range (channels):       
                            current_output = top_output[n][c][lh][lw]
                            top_output[n][c][lh][lw] = current_output + alpha * bottom_values[n][i][c]

    return top_alpha_splats, top_output


def splat_forward_cpu( input, coordinates, values,
                              sigma, soft_normalize= True) :
    
    nr_imgs = input.size(0)
    nr_points = coordinates.size(1)
    nr_channels = input.size(1)
    top_count = nr_imgs * nr_points
    height = input.size(2)
    width = input.size(3)
    alpha_splats = torch.zeros([nr_imgs, height, width], device=values.device)
    output = input.clone()

    alpha_splats, output = SplatForward(
        bottom_coordinates = coordinates ,
        bottom_values =values ,
        bottom_sigma = sigma,
        top_alpha_splats = alpha_splats,
        top_output = output,
        num_points = nr_points,
        channels = nr_channels,
        height = height ,
        width = width,

    )

    alpha_splats = alpha_splats.view(nr_imgs, 1, height, width)

    if soft_normalize:        
        alpha_splats = alpha_splats.clamp(1.0)

    output = output / (alpha_splats + 1e-8)

    return output




