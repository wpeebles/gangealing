import torch
from torchvision.utils import make_grid, save_image
import numpy as np
from PIL import Image, ImageColor
import moviepy.editor
import plotly.graph_objects as go
import plotly.colors
from utils.splat2d_cuda import splat2d
from utils.laplacian_blending import LaplacianBlender
from tqdm import tqdm
import ray
import os

# These are the colorscales for visualizing correspondences in difference clusters.
# The i-th entry corresponds to the i-th cluster. We don't use these for standard, unimodal congealing.
CLUSTER_COLORSCALES = ['plasma', 'plotly3', 'viridis', 'cividis']


def get_colorscale(cluster=None):
    if cluster is None or cluster >= len(CLUSTER_COLORSCALES):
        return 'turbo'  # default colorscale for unimodal congealing
    else:
        return CLUSTER_COLORSCALES[cluster]


def normalize(images, amin=None, amax=None, inplace=False):
    assert images.dim() == 4
    if not inplace:
        images = images.clone()
    if amin is None or amax is None:
        amin, amax = images.amin(dim=(1, 2, 3), keepdims=True), images.amax(dim=(1, 2, 3), keepdims=True)
    else:
        amin, amax = torch.tensor(amin, device=images.device), torch.tensor(amax, device=images.device)
        images = images.clamp_(amin, amax)
    images = images.sub_(amin).div_(torch.maximum(amax - amin, torch.tensor(1e-5, device=images.device)))
    return images


def images2grid(images, **grid_kwargs):
    # images should be (N, C, H, W)
    grid = make_grid(images, **grid_kwargs)
    out = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return out


def load_pil(path, resolution=None):
    img = Image.open(path)
    if resolution is not None:
        img = img.resize((resolution, resolution), Image.LANCZOS)
    img = torch.tensor(np.asarray(img), device='cuda', dtype=torch.float).unsqueeze_(0).permute(0, 3, 1, 2)
    img = img.div(255.0).add(-0.5).mul(2)  # [-1, 1]
    return img  # (1, C, H, W)


def save_video(frames, fps, out_path, filenames=False, codec='libx264', input_is_tensor=False, apply_normalize=True):
    if input_is_tensor:  # (T, C, H, W) in [-1, 1] --> length-T list of (H, W, C) uint8 numpy array in [0, 255]
        if apply_normalize:
            frames = normalize(frames, -1, 1, inplace=True).mul_(255)
        frames = frames.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        frames = [frame for frame in frames]

    duration = len(frames) / fps
    frames = frames[::-1]
    # Pad the end of the video:
    frames.append(frames[-1])
    frames.append(frames[-1])

    def make_frame(t):
        out = frames.pop()
        if filenames:
            out = np.asarray(Image.open(out).convert('RGB')).astype(np.uint8)
        return out

    video = moviepy.editor.VideoClip(make_frame, duration=duration)
    video.write_videofile(out_path, fps=fps, codec=codec, bitrate='50M')


@torch.inference_mode()
def load_dense_label(path, resolution=None, load_colors=False, device='cuda'):
    """
    This function loads an RGBA image and returns the coordinates of pixels that have a non-zero alpha channel value.
    For augmented reality applications, this function can also return the RGB colors of the image (load_colors=True).
    :param path: Path to the RGBA image file
    :param resolution: Resolution to resize the RGBA image to (default: no resizing)
    :param load_colors: If True, returns (1, P, 3) tensor of RGB values in range [-1, 1] (P = number of coordinates)
    :param device: Device to load points and colors to
    :return: (1, P, 2) tensor of pixel coordinates, (1, P, 3) tensor of corresponding RGB colors, (1, P, 1) tensor of
              corresponding non-zero alpha channel values. The pixel coordinates are stored in (x, y) format and are
              integers in the range [0, W-1] (x coordinates) or [0, Y-1] (y coordinates). RGB values are in [-1, 1] and
              alpha channel values are in [0, 1].
    """
    label = torch.from_numpy(np.asarray(Image.open(path))).to(device)  # (H, W, 4) RGBA format
    label = label.permute(2, 0, 1).unsqueeze_(0)  # (1, 4, H, W)
    if resolution is not None and resolution != label.size(0):  # optionally resize the label:
        label = torch.nn.functional.interpolate(label.float(), scale_factor=resolution / label.size(2), mode='bilinear')
    assert label.size(1) == 4
    i, j = torch.where(label[0, 3] > 0)  # i indexes height, j indexes width
    points = torch.stack([j, i], -1)  # (P, 2); points are stored in (x, y) format
    points = points.unsqueeze_(0)  # (1, P, 2)
    if load_colors:
        image = label.float().div_(255.0)  # (1, 4, H, W)
        alpha_channel = image[:, 3:4, i, j].permute(0, 2, 1)  # (1, P, 1), [0, 1]
        colors = image[:, :3, i, j].add(-0.5).mul(2.0).permute(0, 2, 1)  # (1, P, 3), [-1, 1]
    else:
        alpha_channel = torch.ones(1, points.size(1), 1, device=device, dtype=torch.float)
        colors = None
    return points, colors, alpha_channel


@torch.inference_mode()
def load_cluster_dense_labels(path, num_clusters, resolution=None, load_colors=False, device='cuda'):
    assert 'cluster0' in path
    points_per_cluster = []  # will be a list of (1, P_i, 2) tensors
    colors_per_cluster = []  # will be a list of (1, P_i, 3) tensors
    alphas_per_cluster = []  # will be a list of (1, P_i, 1) tensors
    for i in range(num_clusters):
        path = path.replace(f'cluster{max(i - 1, 0)}', f'cluster{i}')
        points, colors, alpha_channel = load_dense_label(path, resolution, load_colors, device)
        points_per_cluster.append(points)
        colors_per_cluster.append(colors)
        alphas_per_cluster.append(alpha_channel)
    return points_per_cluster, colors_per_cluster, alphas_per_cluster


def get_plotly_colors(num_points, colorscale):
    color_steps = torch.linspace(start=0, end=1, steps=num_points).tolist()
    colors = get_color(colorscale, color_steps)
    colors = [plotly.colors.unlabel_rgb(color) for color in colors]
    colors = torch.tensor(colors, dtype=torch.float, device='cuda').view(1, num_points, 3)
    colors = colors.div(255.0).add(-0.5).mul(2)  # Map [0, 255] RGB colors to [-1, 1]
    return colors  # (1, P, 3)


@torch.inference_mode()
def splat_points(images, points, sigma, opacity, colorscale='turbo', colors=None, alpha_channel=None,
                 blend_alg='alpha'):
    """
    Highly efficient GPU-based splatting algorithm. This function is a wrapper for Splat2D to overlay points on images.
    For highest performance, use the colors argument directly instead of colorscale.

    images: (N, C, H, W) tensor in [-1, +1]
    points: (N, P, 2) tensor with values in [0, resolution - 1] (can be sub-pixel/non-integer coordinates)
             Can also be (N, K, P, 2) tensor, in which case points[:, i] gets a unique colorscale
    sigma: either float or (N,) tensor with values > 0, controls the size of the splatted points
    opacity: float in [0, 1], controls the opacity of the splatted points
    colorscale: [Optional] str (or length-K list of str if points is size (N, K, P, 2)) indicating the Plotly colorscale
                 to visualize points with
    colors: [Optional] (N, P, 3) tensor (or (N, K*P, 3)). If specified, colorscale will be ignored. Computing the colorscale
            often takes several orders of magnitude longer than the GPU-based splatting, so pre-computing
            the colors and passing them here instead of using the colorscale argument can provide a significant
            speed-up.
    alpha_channel: [Optional] (N, P, 1) tensor (or (N, K*P, 1)). If specified, colors will be blended into the output
                    image based on the opacity values in alpha_channel (between 0 and 1).
    blend_alg: [Optiona] str. Specifies the blending algorithm to use when merging points into images.
                              Can use alpha compositing ('alpha'), Laplacian Pyramid Blending ('laplacian')
                              or a more conservative version of Laplacian Blending ('laplacian_light')
    :return (N, C, H, W) tensor in [-1, +1] with points splatted onto images
    """
    assert images.dim() == 4  # (N, C, H, W)
    assert points.dim() == 3 or points.dim() == 4  # (N, P, 2) or (N, K, P, 2)
    batch_size = images.size(0)
    if points.dim() == 4:  # each index in the second dimension gets a unique colorscale
        num_points = points.size(2)
        points = points.reshape(points.size(0), points.size(1) * points.size(2), 2)  # (N, K*P, 2)
        if colors is None:
            if isinstance(colorscale, str):
                colorscale = [colorscale]
            assert len(colorscale) == points.size(1)
            colors = torch.cat([get_plotly_colors(num_points, c) for c in colorscale], 1)  # (1, K*P, 3)
            colors = colors.repeat(batch_size, 1, 1)  # (N, K*P, 3)
    elif colors is None:
        num_points = points.size(1)
        if isinstance(colorscale, str):  # All batch elements use the same colorscale
            colors = get_plotly_colors(points.size(1), colorscale).repeat(batch_size, 1, 1)  # (N, P, 3)
        else:  # Each batch element uses its own colorscale
            assert len(colorscale) == batch_size
            colors = torch.cat([get_plotly_colors(num_points, c) for c in colorscale], 0)
    if alpha_channel is None:
        alpha_channel = torch.ones(batch_size, points.size(1), 1, device='cuda')
    if isinstance(sigma, (float, int)):
        sigma = torch.tensor(sigma, device='cuda', dtype=torch.float).view(1).repeat(batch_size)
    blank_img = torch.zeros(batch_size, images.size(1), images.size(2), images.size(3), device='cuda')
    blank_mask = torch.zeros(batch_size, 1, images.size(2), images.size(3), device='cuda')
    prop_obj_img = splat2d(blank_img, points, colors, sigma, False)  # (N, C, H, W)
    prop_mask_img = splat2d(blank_mask, points, alpha_channel, sigma, True) * opacity  # (N, 1, H, W)
    if blend_alg == 'alpha':
        out = prop_mask_img * prop_obj_img + (1 - prop_mask_img) * images  # basic alpha-composite
    elif blend_alg == 'laplacian':
        blender = LaplacianBlender().to(images.device)
        out = blender(images, prop_obj_img, prop_mask_img)
    elif blend_alg == 'laplacian_light':
        blender = LaplacianBlender(levels=3, gaussian_kernel_size=11, gaussian_sigma=0.5).to(images.device)
        out = blender(images, prop_obj_img, prop_mask_img)
    return out


def batch_overlay(images, points, radii, out_path, unique_color=False, size=10, normalize=True,
                  opacity=1.0, colorscale=None, range=None, **marker_kwargs):
    os.makedirs(out_path, exist_ok=True)
    futures = []
    paths = []
    for i, (image, point) in enumerate(tqdm(zip(images, points), total=len(images))):
        out_path_i = f'{out_path}/{i}.png'
        future = overlay_points(image, point, radii, out_path_i, unique_color=unique_color, size=size,
                                normalize=normalize, opacity=opacity, colorscale=colorscale,
                                range=range, **marker_kwargs)
        paths.append(out_path_i)
        futures.append(future)
    # ray.get(futures)
    out = []
    for path in paths:
        out.append(torch.from_numpy(np.asarray(Image.open(path).convert('RGB'))).float())
    out = torch.stack(out).permute(0, 3, 1, 2)
    return out


@ray.remote
def overlay_points_parallel(image, points, radii, out_path, unique_color=False, size=10, normalize=True,
                            opacity=1.0, colorscale=None, range=None, **marker_kwargs):
    overlay_points(image, points, radii, out_path, unique_color, size, normalize, opacity, colorscale, range,
                   **marker_kwargs)


def overlay_points(image, points, radii, out_path, unique_color=False, size=10, normalize=True, opacity=1.0,
                   colorscale=None, range=None):
    # This function takes in a single image in (C, H, W) format (or a PIL Image directly) and overlays the points
    # on it. This function can be pretty slow, so it's recommended to invoke this function with overlay_points_parallel
    # to take advantage of parallelization via ray.
    if normalize:  # Load image as PIL.Image to work with plotly:
        assert image.dim() == 3
        img_size = image.size(-1)
        ndarr = images2grid(image, padding=0, normalize=True, range=range)
        pil_img = Image.fromarray(ndarr)
    else:  # Image is already in PIL.Image format:
        img_size, _ = image.size
        pil_img = image
    fig = go.Figure()
    if not isinstance(points, list):
        points = [points]
    if not isinstance(colorscale, list):
        colorscale = [colorscale]
    for kps, colorscale in zip(points, colorscale):
        assert kps.dim() == 2
        kps = kps.cpu().numpy()
        marker_kwargs = {'colorscale': colorscale} if colorscale is not None else {}
        if radii is not None:
            radii = radii.cpu().numpy()
            fig.add_trace(go.Scatter(x=kps[:, 0], y=img_size - kps[:, 1], mode="markers", marker=dict(size=size+radii, color=3, opacity=0.15, **marker_kwargs)))
        if unique_color:
            c = np.arange(0, kps.shape[0])
        else:
            c = 0
        fig.add_trace(go.Scatter(x=kps[:, 0], y=img_size - kps[:, 1], mode="markers", marker=dict(size=size, color=c, opacity=opacity, **marker_kwargs)))  # Key points

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_size]
    )
    fig.update_yaxes(
        visible=False,
        range=[0, img_size],
        scaleanchor="x"
    )
    fig.add_layout_image(
        dict(source=pil_img,
             xref="x",
             yref="y",
             x=0,
             y=img_size,
             sizex=img_size,
             sizey=img_size,
             sizing="stretch",
             opacity=1,
             layer="below")
    )
    fig.update_layout(
        width=img_size,
        height=img_size,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False
    )
    fig.write_image(out_path)


def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        intermediate_colors = [get_continuous_color(colorscale, x) for x in loc]
        return intermediate_colors
    return get_continuous_color(colorscale, loc)


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    intermediate_color = plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )
    return intermediate_color
