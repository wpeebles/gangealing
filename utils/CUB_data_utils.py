import numpy as np

# Functions below are mostly from the ACSM codebase (with light modifications): https://github.com/nileshkulkarni/acsm


def perturb_bbox(bbox, pf=0, jf=0):
    '''
    Jitters and pads the input bbox.

    Args:
        bbox: Zero-indexed tight bbox.
        pf: padding fraction.
        jf: jittering fraction.
    Returns:
        pet_bbox: Jittered and padded box. Might have -ve or out-of-image coordinates
    '''
    pet_bbox = [coord for coord in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    pet_bbox[0] -= (pf*bwidth) + (1-2*np.random.random())*jf*bwidth
    pet_bbox[1] -= (pf*bheight) + (1-2*np.random.random())*jf*bheight
    pet_bbox[2] += (pf*bwidth) + (1-2*np.random.random())*jf*bwidth
    pet_bbox[3] += (pf*bheight) + (1-2*np.random.random())*jf*bheight

    return pet_bbox


def python2_round(n):
    # ACSM is Python2; for parity, we use Python2 rounding in these utils.
    # https://stackoverflow.com/a/33019948
    from decimal import localcontext, Decimal, ROUND_HALF_UP
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP
        rounded = Decimal(n).to_integral_value()
    return rounded


def square_bbox(bbox, py2_round=True):
    '''
    Converts a bbox to have a square shape by increasing size along non-max dimension.
    '''
    round_fn = python2_round if py2_round else round
    sq_bbox = [int(round_fn(coord)) for coord in bbox]
    bwidth = sq_bbox[2] - sq_bbox[0] + 1
    bheight = sq_bbox[3] - sq_bbox[1] + 1
    maxdim = float(max(bwidth, bheight))

    dw_b_2 = int(round_fn((maxdim - bwidth) / 2.0))
    dh_b_2 = int(round_fn((maxdim - bheight) / 2.0))

    sq_bbox[0] -= dw_b_2
    sq_bbox[1] -= dh_b_2
    sq_bbox[2] = sq_bbox[0] + maxdim - 1
    sq_bbox[3] = sq_bbox[1] + maxdim - 1

    return sq_bbox


def acsm_crop(img, bbox, bgval=0, border=True, py2_round=True):
    '''
    Crops a region from the image corresponding to the bbox.
    If some regions specified go outside the image boundaries, the pixel values are set to bgval.

    Args:
        img: image to crop
        bbox: bounding box to crop
        bgval: default background for regions outside image
    '''
    round_fn = python2_round if py2_round else round
    bbox = [int(round_fn(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]

    nc = 1 if len(im_shape) < 3 else im_shape[2]

    img_out = np.ones((bheight, bwidth, nc), dtype=np.uint8) * bgval
    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2] + 1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3] + 1)

    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg
    if border:
        img_in = img_out = img[y_min_src:y_max_src, x_min_src:x_max_src, :]
        left_pad = x_min_trg
        right_pad = bwidth - x_max_trg
        up_pad = y_min_trg
        down_pad = bheight - y_max_trg
        try:
            img_out = np.pad(img_out, mode='edge', pad_width=[(up_pad, down_pad), (left_pad, right_pad), (0, 0)])
            assert ((img_out[y_min_trg:y_max_trg, x_min_trg:x_max_trg, :] - img_in) ** 2).sum() == 0
            assert img_out.shape[0] == img_out.shape[1]
        except ValueError:
            print(f'crop_shape: {img_out.shape}, pad: {(up_pad, down_pad, left_pad, right_pad)}, trg: {(y_min_trg, y_max_trg, x_min_trg, x_max_trg)}, box: {(bheight, bwidth)}, img_shape: {im_shape}')
            exit()
        # assert img_out.shape == (256, 256, 3), f'got {img_out.shape}, {x_min_trg}, {x_max_trg}, {y_min_trg}, {y_max_trg}'
    else:
        img_out[y_min_trg:y_max_trg, x_min_trg:x_max_trg, :] = img[y_min_src:y_max_src, x_min_src:x_max_src, :]
    return img_out
