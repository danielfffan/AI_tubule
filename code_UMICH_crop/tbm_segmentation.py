

import numpy as np
import cv2
import glob
from tqdm.autonotebook import tqdm
from skimage import morphology
from skimage.color import rgb2hed, hed2rgb
from PIL import Image, ImageFilter
import argparse

def rgb_gray(rgb):
    return np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])

parser = argparse.ArgumentParser(description='Generate mask images from IHC')

parser.add_argument('-p_', '--threshold_1_step2', help="The threshold for the first binary process", default=180, type=int)
parser.add_argument('-p2_', '--threshold_2_step2', help="The threshold for the second binary process", default=90, type=int)
parser.add_argument('-l_', '--blur_step2', help="The threshold for the blur process.", default= 2, type=int)
parser.add_argument('-ob', '--ob_step2', help="The threshold for remove small objects.", default= 100 , type=int)
parser.add_argument('-ho', '--ho_step2', help="The threshold for remove small holes.", default=50, type=int)
parser.add_argument('-n', '--datasetname', help="dataset name", default='Neptune_data', type=str)

args_ = parser.parse_args()

print(f"args: {args_}")

th_1_ = args_.threshold_1_step2
th_2_ = args_.threshold_2_step2
blur_ = args_.blur_step2
ob_ = args_.ob_step2
ho_ = args_.ho_step2

dataset = args_.datasetname
pas_files = glob.glob('/Volumes/data/UMich/thickened_TBM_test/cortex_image/*.png')

print(pas_files)
for i in range(len(pas_files)):
    # for injured TBM
    pas_fname = pas_files[i]
    print(pas_fname)
    ihc_rgb = cv2.imread(pas_fname)
    # Separate the stains from the IHC image
    ihc_hed = rgb2hed(ihc_rgb)

    tbm_segmentation_mask = pas_fname.replace('cortex_image/', 'tbm_IP/')
    # cv2.imwrite('ihc_hed.png',ihc_hed)

    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    # ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    eosin = ihc_e * 255

    ret_, thresh_ = cv2.threshold(eosin, th_1_, 255, cv2.THRESH_BINARY)
    # cv2.imwrite(tbm_segmentation_mask, eosin)
    #
    img_mean = cv2.blur(thresh_, (blur_, blur_))
    # cv2.imwrite('blur.png', img_mean)
    thresh0_ = rgb_gray(img_mean)
    # cv2.imwrite('gray.png', thresh0_)

    img_ = thresh0_
    for x in range(img_.shape[0]):
        for y in range(img_.shape[1]):
            px = img_[x, y]
            if px > 220:
                img_[x, y] = 0
            else:
                img_[x, y] = 255
    # cv2.imwrite('binary.png', img_)

    arr_ = img_ > 0
    img_ = morphology.remove_small_objects(arr_, min_size=ob_)
    # print("sss", ob_)
    # cleaned = morphology.remove_small_holes(img, area_threshold=1000, connectivity=1, in_place=False)
    img_ = img_ + 0
    img_ = img_ * 255
    # cv2.imwrite(f'mask_step2/remove_small_obj_{ob_}.png',img_)

    arr_ = img_ > 0
    cleaned_ = morphology.remove_small_holes(arr_, area_threshold=ho_, connectivity=1)
    cleaned_ = cleaned_ + 0
    cleaned_ = cleaned_ * 255

    #for healthy tbm
    # tubule_seg = pas_fname.replace('PAS/','tub_seg/').replace('.png','_mask.png')
    # mask = cv2.imread(tubule_seg)
    # kernel_ero = np.ones((8, 8), np.uint8)
    # img_dilation = cv2.erode(mask, kernel_ero, iterations=1)
    # normal_tbm = (img_dilation - mask)[:,:,0]
    # #combine together
    # tbm = normal_tbm+cleaned_
    cv2.imwrite(tbm_segmentation_mask, cleaned_)


