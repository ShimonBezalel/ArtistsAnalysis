import numpy as np
from skimage import feature
from skimage import io
from skimage import color
import traceback
from scipy.signal import (convolve, convolve2d)

import os

# Directory and file names
artist_dir          = "artist_data"
desc                = "descriptor"
desc_poi            = 'texture_patch_desc'
desc_canny          = 'canny_desc'
desc_automated      = "auto_desc"
desc_frequency      = 'freq_desc'
hist                = "hist_desc"

color_file          = 'paint'
patches_dir_name    = 'patches3'
auto_dir_name       = "automated"


# constants
PATCH_DIM           = 128
batch_size          = 10
BINS                = 256
GUASSIAN_KERNEL     = np.array([0.06136, 0.24477, 0.38774, 0.24477, 0.06136])[np.newaxis, ...]


def describe_artist(artist_path, canny=False, canny2=False, frequency=False, color_hist=False, names=False):
    """
    Given some stradegy, describes all the patches for a given artist and saves in descritor folder under unique name.
    :param artist_path: Path for artist specific library
    :param canny:  bool : using canny stradegy
    :param canny2: bool: using canny stradegy with sigma = 3
    :param frequency: bool: using fft stradegy
    :param color_hist: bool: using hue histogram stradegy
    :param names: bool: saving names of images and patches corresponding to descriptors
    """
    patch_dir = os.path.join(artist_path,  patches_dir_name)
    desc_dir = os.path.join(artist_path, desc)
    auto_dir = os.path.join(patch_dir, auto_dir_name)
    source = auto_dir

    try:
        os.mkdir(desc_dir)
    except:
        pass

    if color_hist:
        choice_shape = (batch_size, BINS)
        choice_type = np.float64
    elif frequency:
        choice_shape = (batch_size, PATCH_DIM, PATCH_DIM)
        choice_type = np.complex64
    elif names:
        choice_shape = (0, )
        choice_type = np.str
    elif canny or canny2:
        choice_shape = (batch_size, PATCH_DIM, PATCH_DIM)
        choice_type = np.bool
    else:
        choice_shape = (batch_size, PATCH_DIM, PATCH_DIM)
        choice_type = np.float64


    artist_desriptors = np.zeros(shape=choice_shape).astype(choice_type)
    for patch in filter(lambda file: file.endswith(".png"), os.listdir(source)):
        full_path = os.path.join(source, patch)
        try:
            patch_orig = io.imread(fname=full_path)[..., :3]
            hsv_im = color.rgb2hsv(patch_orig)
            hues = hsv_im[..., 0]
            grays = hsv_im[..., 2]
        except Exception as e:
            print("Could not open patch file {}".format(patch))
            print(e)
            continue

        if names:
            descriptors = np.tile(np.array(full_path), batch_size).astype(choice_type)
        else:
            patch_color = hues if color_hist else grays

            xs = np.random.choice(np.arange(0, patch_color.shape[1] - PATCH_DIM), size=batch_size).astype(np.int)
            ys = np.random.choice(np.arange(0, patch_color.shape[0] - PATCH_DIM), size=batch_size).astype(np.int)

            descriptors = np.zeros(shape=choice_shape).astype(choice_type)

            for i in range(batch_size):
                if canny:
                    descriptors[i] = feature.canny(patch_color[ys[i]:ys[i] + PATCH_DIM, xs[i]:xs[i] + PATCH_DIM])
                elif canny2:
                    descriptors[i] = feature.canny(patch_color[ys[i]:ys[i] + PATCH_DIM, xs[i]:xs[i] + PATCH_DIM], sigma=3)
                elif frequency:
                    descriptors[i] = np.fft.fft2(patch_color[ys[i]:ys[i] + PATCH_DIM, xs[i]:xs[i] + PATCH_DIM])
                elif color_hist:
                    hues_hist = np.histogram(a=(patch_color[ys[i]:ys[i] + PATCH_DIM, xs[i]:xs[i] + PATCH_DIM]),
                                                  bins=choice_shape[1])[0][np.newaxis, ...]
                    descriptors[i] = convolve2d(hues_hist, GUASSIAN_KERNEL, mode='same', boundary='wrap')
                else: # no chnages
                    descriptors[i] = patch_color[ys[i]:ys[i] + PATCH_DIM, xs[i]:xs[i] + PATCH_DIM]

        artist_desriptors = np.append(artist_desriptors, descriptors, axis=0)

    # Generate unique name for this descriptor
    target = ""
    target += desc_canny if canny else ""
    target += desc_canny + '2' if canny2 else ""
    target += desc_frequency if frequency else ""
    target += hist if color_hist else ""
    target += desc_automated
    target = "names" if names else target


    np.save(os.path.join(desc_dir, target), arr=artist_desriptors)


if __name__ == '__main__':
    fil = lambda artist: artist in {"Pablo Picasso"}
    for artist_path in filter(fil, os.listdir(artist_dir)):
        try:
            describe_artist(os.path.join(artist_dir, artist_path, color_file), canny=True)
        except Exception as e:
            print("could not describe %d".format(artist_path))
            traceback.print_exc()
            continue

