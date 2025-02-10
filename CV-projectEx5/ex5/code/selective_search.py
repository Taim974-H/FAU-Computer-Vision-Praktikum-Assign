# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np


def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    # if im_orig.dtype != np.float32:
    #     im_orig = skimage.util.img_as_float32(im_orig)

    segment_labels = skimage.segmentation.felzenszwalb(
        im_orig, scale=scale, sigma=sigma, min_size=min_size
    )
    # image_mask = np.zeros(segment_labels.shape)
    # for i, label in enumerate(np.unique(segment_labels)):
    #     image_mask[segment_labels == label] = i
    # im_seg = np.append(im_orig, np.atleast_3d(image_mask), axis=2)
    
    # Use pre-allocation instead of zeros
    im_seg = np.empty((im_orig.shape[0], im_orig.shape[1], im_orig.shape[2] + 1))
    im_seg[...,:3] = im_orig
    im_seg[...,3] = segment_labels
    
    return im_seg


def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###
    color_sim_score = np.sum(np.minimum(r1['hist_c'], r2['hist_c']))
    return color_sim_score


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    ### YOUR CODE HERE ###
    texture_sim_score = np.sum(np.minimum(r1['hist_t'], r2['hist_t']))
    return texture_sim_score


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    ### YOUR CODE HERE ###
    merged_size = r1['size'] + r2['size']
    size_sim_score = 1 - (merged_size / imsize)
    return size_sim_score


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###
    # compare coords to find the bounding box that contains both regions
    min_x = min(r1['min_x'], r2['min_x'])
    min_y = min(r1['min_y'], r2['min_y'])
    max_x = max(r1['max_x'], r2['max_x'])
    max_y = max(r1['max_y'], r2['max_y'])
    
    # Calculate size of combined bounding box
    combined_box_size = (max_x - min_x) * (max_y - min_y)
    
    # Calculate fill similarity: 1 - (combined_box_size / total_image_size)
    fill_sim_score = 1 - (combined_box_size / imsize)
    
    return fill_sim_score


def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))


def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """

    BINS = 25
    hist = np.zeros(BINS * img.shape[2])
    
    # Calculate histogram for each channel
    for channel in range(img.shape[2]):
        channel_hist, _ = np.histogram(
            img[:, :, channel].ravel(),
            bins=BINS,
            range=(0, 1),
            density=True
        )
        hist[channel * BINS:(channel + 1) * BINS] = channel_hist
    
    return hist

def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    
    # # Convert to uint8 only if needed
    # if img.dtype != np.uint8:
    #     img = skimage.util.img_as_ubyte(img)
    
    points = 8
    radius = 1.0
    method = 'uniform'
    
    for channel in range(img.shape[2]):
        ret[:, :, channel] = skimage.feature.local_binary_pattern(
            img[:, :, channel],
            P=points,
            R=radius,
            method=method
        )
    return ret


def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.zeros(BINS * img.shape[2]) # pre-allocate histogram array
    # Calculate texture gradient once
    texture_grad = calc_texture_gradient(img)
    
    # Calculate histogram for each channel
    for channel in range(img.shape[2]):
        channel_hist, _ = np.histogram(
            texture_grad[:, :, channel].ravel(),
            bins=BINS,
            density=True
        )
        hist[channel * BINS:(channel + 1) * BINS] = channel_hist
    
    # L1 normalization
    if np.sum(hist) != 0:
        hist = hist / np.sum(hist)
    
    return hist

def extract_regions(img):
    """
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    """
    R = {}
    
    # HSV conversion
    hsv_img = skimage.color.rgb2hsv(img[:, :, :3])
    
    # Get labels and unique labels
    labels = img[:, :, 3]
    unique_labels = np.unique(labels).astype(int)
    
    for label in unique_labels:
        # Create mask for current label
        mask = labels == label
        y, x = np.nonzero(mask)
        
        if len(y) == 0:  # Skip empty regions
            continue
            
        # Calculate bounding box
        min_y, max_y = y.min(), y.max()
        min_x, max_x = x.min(), x.max()
        
        # Extract region data
        region = np.zeros_like(img[:, :, :3])
        region[mask] = img[:, :, :3][mask]
        
        # Extract HSV data
        hsv_region = np.zeros_like(region)
        hsv_region[mask] = hsv_img[mask]
        
        R[label] = {
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y,
            'size': len(y),
            'labels': [label],
            'rect': (min_y, min_x, max_x - min_x, max_y - min_y),
            'hist_c': calc_colour_hist(hsv_region),
            'hist_t': calc_texture_hist(region)
        }
    
    return R

def extract_neighbours(regions): #regions is a dictionary of all regions
    #check if two regions intersect by comparing their bounding boxes
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###

    # Get list of all region
    all_regions_keys = list(regions.keys())

    tracked_regions = []
    # Go through each region
    for region in all_regions_keys:
        # Add current region to checked list
        tracked_regions.append(region)
        
        # Look at all other regions we haven't checked yet
        for other_region in all_regions_keys:

            if other_region in tracked_regions:
                continue  # Skip if we've already checked this pair
            #else:    
            # Get both regions
            region1 = regions[region]
            region2 = regions[other_region]
            
            # If they intersect, add them as neighbors
            if intersect(region1, region2):
                neighbours.append((
                    (region, region1),
                    (other_region, region2)
                ))
    
    return neighbours


def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    ### YOUR CODE HERE

    # Find new bounding box coordinates
    rt['min_x'] = min(r1['min_x'], r2['min_x'])
    rt['min_y'] = min(r1['min_y'], r2['min_y'])
    rt['max_x'] = max(r1['max_x'], r2['max_x'])
    rt['max_y'] = max(r1['max_y'], r2['max_y'])
    
    # Set size and combine labels
    rt['size'] = new_size
    rt['labels'] = r1['labels'] + r2['labels']
    
    width = rt['max_x'] - rt['min_x']
    height = rt['max_y'] - rt['min_y']
    Bounding_box = (rt['min_x'], rt['min_y'], width, height)
    rt['rect'] = Bounding_box


    # Merge histograms - weighted average based on region sizes
    rt['hist_c'] = (
        (r1['hist_c'] * r1['size'] + r2['hist_c'] * r2['size']) / new_size
    )
    
    rt['hist_t'] = (
        (r1['hist_t'] * r1['size'] + r2['hist_t'] * r2['size']) / new_size
    )
    
    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        ### YOUR CODE HERE ###
        delete_keys = [] # keys are the region pairs to be removed
        for regionPair, value in S.items():
            if (i in regionPair) or (j in regionPair):
                delete_keys.append(regionPair)

        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###
        for regionPair in delete_keys:
            del S[regionPair]

        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###

        for regionPair in delete_keys:
            # skip the direct pair between i and j since they're now merged
            if regionPair in {(i, j), (j, i)}:
                continue

            # determine which region in the pair is being connected to the new region
            region_a, region_b = regionPair
            n = (
                region_b if region_a in {i, j}  # If first element is being merged
                else region_a                   # If second element is being merged
            )
            
            new_similarity = calc_sim(
                R[t], # The new merged region
                R[n], # The neighboring region
                imsize
            )
            S[(t, n)] = new_similarity

    # Task 8: Generating the final regions from R
    regions = []
    ### YOUR CODE HERE ###
    for k, r in R.items():
        regions.append({
            'rect': (
                r['min_x'],
                r['min_y'],
                r['max_x'] - r['min_x'],
                r['max_y'] - r['min_y']
            ),
            'labels': r['labels'],
            'size': r['size']
            
        })

    return image, regions


