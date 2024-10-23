import numpy as np

# from geojson import Polygon
import json
import os
from shapely.geometry import Polygon
import openslide
import staintools
import cv2
import math
import glob
import os
import skimage.io as skio
import staintools
import pandas as pd

import sys
from tqdm import tqdm


def Nrotate(angle, valuex, valuey, pointx, pointy):
    angle = (angle / 180) * math.pi
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    nRotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return (int(nRotatex), int(nRotatey))
def Srotate(angle, valuex, valuey, pointx, pointy):
    angle = (angle / 180) * math.pi
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex - pointx) * math.cos(angle) + (valuey - pointy) * math.sin(angle) + pointx
    sRotatey = (valuey - pointy) * math.cos(angle) - (valuex - pointx) * math.sin(angle) + pointy
    return (int(sRotatex), int(sRotatey))
def rotatecordiate(angle, rectboxs, pointx, pointy):
    output = []
    for rectbox in rectboxs:
        if angle > 0:
            output.append(Srotate(angle, rectbox[0], rectbox[1], pointx, pointy))
        else:
            output.append(Nrotate(-angle, rectbox[0], rectbox[1], pointx, pointy))
    return output
def imagecrop(image, box):
    xs = [x[1] for x in box]
    ys = [x[0] for x in box]
    print(xs)
    print(min(xs), max(xs), min(ys), max(ys))
    cropimage = image[min(xs):max(xs), min(ys):max(ys)]
    print(cropimage.shape)
    # cv2.imwrite('cropimage.png', cropimage)
    return cropimage
def check_back_ground(roi):
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img,50,255,0)
    rgb_0 = len(thresh[thresh == 0])
    print(f"black pixel numbers: {rgb_0}")
    if (rgb_0 > 500):
        return True
    else:
        return False
def stain_normalization(roi_file_names):
    roi_sn_file_names = []
    target = staintools.read_image("./16-064_15441_33189.png")
    target = staintools.LuminosityStandardizer.standardize(target)

    for i in range(len(roi_file_names)):
        roi_name = roi_file_names[i]
        to_transforms = roi_name
        print("************This is for stain normalization***************")
        name_sn = roi_name.replace("ROI_rmbg/", "ROI_sn/")
        print(name_sn)
        roi_sn_file_names.append(name_sn)
        flag = os.path.exists(name_sn)
        if (flag == False):
            to_transforms = staintools.read_image(to_transforms)

            # target = staintools.LuminosityStandardizer.standardize(target)
            to_transforms = staintools.LuminosityStandardizer.standardize(to_transforms)

            # Stain normalize
            normalizer = staintools.StainNormalizer(method='vahadane')
            normalizer.fit(target)
            transformed = normalizer.transform(to_transforms)
            print('Writing the file...')
            cv2.imwrite(name_sn, transformed)
            print("Done!")
        else:
            print("File already exists!")

    return roi_sn_file_names
def bounding_box(points):
    top_left_x, top_left_y = float('inf'), float('inf')
    bot_right_x, bot_right_y = float('-inf'), float('-inf')
    for x, y in points:
        top_left_x = min(top_left_x, x)
        top_left_y = min(top_left_y, y)
        bot_right_x = max(bot_right_x, x)
        bot_right_y = max(bot_right_y, y)
    return top_left_x, top_left_y, bot_right_x, bot_right_y
    # return [(top_left_x, top_left_y), (bot_right_x, bot_right_y)]
def rgb_median (roi,mask):
    cv2.bitwise_not(mask, mask, mask=None)
    # mask = mask[:, :, 0]
    bg = cv2.bitwise_and(roi, roi, mask=mask)
    mask_chan1 = bg[:, :, 0]
    mask_chan2 = bg[:, :, 1]
    mask_chan3 = bg[:, :, 2]
    chan_1 = np.median(mask_chan1[np.nonzero(mask_chan1)])
    chan_2 = np.median(mask_chan2[np.nonzero(mask_chan2)])
    chan_3 = np.median(mask_chan3[np.nonzero(mask_chan3)])
    if ((chan_1 < 180) or (chan_2<180) or (chan_3<180)):
        chan_1 = 215
        chan_2 = 215
        chan_3 = 215
    else:
        chan_1 = np.median(mask_chan1[np.nonzero(mask_chan1)])
        chan_2 = np.median(mask_chan2[np.nonzero(mask_chan2)])
        chan_3 = np.median(mask_chan3[np.nonzero(mask_chan3)])
    print(f'median pixel values: {chan_1},{chan_2},{chan_3}')

    return chan_1,chan_2,chan_3
def is_image(I):
    """
    Is I an image.
    """
    if not isinstance(I, np.ndarray):
        return False
    if not I.ndim == 3:
        return False
    return True
def is_uint8_image(I):
    """
    Is I a uint8 image.
    """
    if not is_image(I):
        return False
    if I.dtype != np.uint8:
        return False
    return True
def get_tissue_mask(I, luminosity_threshold):
        # print('Checking if there is tissue...')
        # """
        # Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
        # Typically we use to identify tissue in the image and exclude the bright white background.
        #
        # :param I: RGB uint 8 image.
        # :param luminosity_threshold: Luminosity threshold.
        # :return: Binary mask.
        # """
        # assert is_uint8_image(I), "Image should be RGB uint8."
        # I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
        # L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
        # mask = L < luminosity_threshold
        #
        # # Check it's not empty
        # if mask.sum() == 0:
        #     print("it is no empty!")
        #     return False
        # else:
        #     print('There is tissue!')
        #     return True
        return True
def remove_background(x_min,y_min,points,roi_name):
    points_copy = []
    for i in range(len(points)):
        point_copy = [points[i][0],points[i][1]]
        points_copy.append(point_copy)
    roi_rmbg_name = roi_name.replace('tissue', 'tissue_rmbg')
    roi_cor_mask = roi_name.replace('tissue','cortex_mask')
    print("Doing the remove background process......")
    if(os.path.exists(roi_rmbg_name) == False):
        roi = cv2.imread(roi_name)
        if (get_tissue_mask(roi, luminosity_threshold=0.2)):
            print("Removing the background......")
            points_ = []
            for jj in range(len(points_copy)):
                point = points_copy[jj]
                point[0] = int((point[0] - x_min))
                point[1] = int((point[1] - y_min))
                points_.append(point)
            pts = np.array(points_)
        # print(pts)
            croped = roi.copy()
            mask = np.zeros(croped.shape[:2], np.uint8)
        # mask = cv2.fillPoly(mask, pts =[pts], color=(255,255,255))
            mask = cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.imwrite(roi_cor_mask,mask)
        # rect = cv2.minAreaRect(pts)
        # # print(np.array(points_))
        # box_origin = cv2.boxPoints(rect)
        # M = cv2.getRotationMatrix2D(rect[0], rect[2], 1)
        # dst = cv2.warpAffine(mask, M, (3 * mask.shape[0], 3 *  mask.shape[1]))
        # box = rotatecordiate(rect[2], box_origin, rect[0][0], rect[0][1])
        # mask = imagecrop(dst, np.int0(box))
        # cv2.imwrite(roi_cor_mask, mask)
        # mask =cv2.imread('draw_contour.png')

            roi = cv2.imread(roi_name)
            croped = roi.copy()
        # print(croped.shape)
        # print(mask.shape)

            dst = cv2.bitwise_and(croped, croped, mask=mask)
        #cv2.imwrite('dst.png',dst)
            bg = np.ones_like(croped, np.uint8)
            r, g, b = rgb_median(roi,mask)
            bg[:, :, 0] = bg[:, :, 0] * r
            bg[:, :, 1] = bg[:, :, 1] * g
            bg[:, :, 2] = bg[:, :, 2] * b
        #cv2.imwrite('bg_1.png', bg)
        # cv2.bitwise_not(mask, mask, mask=None)
        #cv2.imwrite('mask.png', mask)
        #cv2.imwrite('bg_update.png',bg)
            bg = cv2.bitwise_and(bg,bg,mask=mask)
        #cv2.imwrite('bg_2.png', bg)
            dst2 = bg + dst
            cv2.imwrite(roi_rmbg_name,dst2)
            print('done!')
    else:
        print('File already exists!')
    return

def minimum_bbox_roi(x_min,y_min,points,roi_name):
    points_ = []
    roi_name_minimum_boundingbox = roi_name.replace('ROI/','ROI_minibox/')
    print("Doing the generating the minimum box process......")
    if (os.path.exists(roi_name_minimum_boundingbox) == False):
        roi = cv2.imread(roi_name)
        print("Generating the minimum box of ROI......")
        for jj in range(len(points)):
            point = points[jj]
            point[0] = int((point[0] - x_min)/2)
            point[1] = int((point[1] - y_min)/2)
            points_.append(point)
        points_ = np.array(points_)
        rect = cv2.minAreaRect(points_)
        # print(np.array(points_))
        box_origin = cv2.boxPoints(rect)
        M = cv2.getRotationMatrix2D(rect[0], rect[2], 1)
        dst = cv2.warpAffine(roi, M, (2 * roi.shape[0], 2 * roi.shape[1]))
        box = rotatecordiate(rect[2], box_origin, rect[0][0], rect[0][1])
        cropimage = imagecrop(dst, np.int0(box))
        cv2.imwrite(roi_name_minimum_boundingbox,cropimage)
        print('done!')
    else:
        print('File already exists!')

    return points_, roi_name_minimum_boundingbox

def ifta_preifta_strore (x_min,y_min,points_cortex,point_ifta,point_preifta,tissue_roi_name):

    polygon_cortex = Polygon(points_cortex)
    roi = cv2.imread(tissue_roi_name)
    croped = roi.copy()
    mask = np.zeros(croped.shape[:2], np.uint8)
    mask_ifta_name = tissue_roi_name.replace('tissue','ifta')
    ifta_num = 0
    for i in range(len(point_ifta)):
        # print(f'ifta_{i}')
        ifta_cnt = point_ifta[i]
        polygon_ifta = Polygon(ifta_cnt)
        condition_ifta_contain = polygon_cortex.contains(polygon_ifta)
        condition_ifta_intersec = polygon_ifta.intersects(polygon_cortex)
        # print(condition_ifta)
        points_ifta = []
        if ((condition_ifta_contain == True) or (condition_ifta_intersec == True)):
            ifta_num += 1
            # print("ifta is inside the cortex")
            for jj in range(len(ifta_cnt)):
                point = ifta_cnt[jj]
                point[0] = int((point[0] - x_min))
                point[1] = int((point[1] - y_min))
                points_ifta.append(point)
                pts = np.array(points_ifta)
                mask = cv2.polylines(mask, [pts], False, (255, 255, 255), 10)
                # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts:
                    cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
                # cv2.fillPoly(mask, [pts], 255)
                # mask = cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.FILLED)
                # mask = cv2.drawContours(mask, [pts] , (255, 255, 255), cv2.FILLED)
    print(f"There are {ifta_num} ifta areas")
    if (ifta_num != 0):
        print('Storing ifta mask......')
    # mask = cv2.fillPoly(mask, pts =[pts], color=(255,255,255))
        cv2.imwrite(mask_ifta_name, mask)
        print('Done!')

    mask = np.zeros(croped.shape[:2], np.uint8)
    mask_preifta_name = tissue_roi_name.replace('tissue', 'pre_ifta')
    preifta_num = 0
    for i in range(len(point_preifta)):
        # print(f'pre_ifta_{i}')
        preifta_cnt = point_preifta[i]
        polygon_preifta = Polygon(preifta_cnt)
        condition_preifta_contain = polygon_cortex.contains(polygon_preifta)
        condition_preifta_intersec = polygon_preifta.intersects(polygon_cortex)
        points_preifta = []
        # print(condition_preifta)
        if ((condition_preifta_contain == True) or(condition_preifta_intersec==True) ):
            preifta_num += 1
            # print("preifta is inside the cortex")
            for jj in range(len(preifta_cnt)):
                point = preifta_cnt[jj]
                point[0] = int((point[0] - x_min))
                point[1] = int((point[1] - y_min))
                points_preifta.append(point)
                pts = np.array(points_preifta)
                mask = cv2.polylines(mask, [pts], False, (255, 255, 255), 10)
                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts:
                    cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
                # cv2.fillPoly(mask, [pts], 255)
                # mask = cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.FILLED)

    print(f"There are {preifta_num} preifta areas")
    if (preifta_num != 0):
        print('Storing preifta mask......')
        cv2.imwrite(mask_preifta_name, mask)
        print('Done!')
    return
# anno_files = glob.glob('IFTA_WSI_label_json/*.json')
# wsi_id_all = []
# cortex_all = []
# cortex_bbox_all = []
# ifta_all = []
# pre_ifta_all = []
#
# cortex_num = []
# cortex_bbox_num = []
# ifta_num = []
# pre_ifta_num = []
# # for file in tqdm(anno_files):
# for n in range(len(anno_files)):
#     cortex = []
#     cortex_bbox = []
#     ifta = []
#     # ifta_bbox = []
#     pre_ifta = []
#     # pre_ifta_bbox=[]
#     file = anno_files[n]
#     print(file)
#     fold = file.split('/')[1]
#     print(fold)
#     wsi_id = fold.split('.json')[0]
#     wsi_id_all.append(wsi_id)
#     # flag_name = os.path.exists(wsi_id)
#     # if (flag_name == False):
#     #     os.mkdir(wsi_id)
#     # #get the wsi file path
#     #
#     # slide = openslide.OpenSlide('0_3312_A_0049953.ndpi')
#
#     with open(file, 'r') as load_f:
#         load_dict = json.load(load_f)
#         for i in range(len(load_dict)):
#             # print(i)
#             info = load_dict[i]
#             # print(info)
#             prop = info['properties']
#             prop_name = prop['classification']['name']
#             print(prop_name)
#             if (prop_name == 'Ignore*'):
#                 print('This is the cortex annotation')
#                 coordinates = info['geometry']['coordinates']
#                 if (info['geometry']['type'] == 'Polygon'):
#                     points = []
#                     #         print(coordinates)
#                     coordinates = coordinates[0]
#                     for j in range(len(coordinates)):
#                         #            print(coordinates[j])
#                         points.append(coordinates[j])
#                     #        print(points)
#                     # return [(top_left_x, top_left_y), (bot_right_x, bot_right_y)]
#                     top_left_x, top_left_y, bot_right_x, bot_right_y = bounding_box(points)
#                     if (((bot_right_x - top_left_x) * (bot_right_y - top_left_y)) > 50000):
#                         cortex_bbox.append([(top_left_x, top_left_y), (bot_right_x, bot_right_y)])
#                         # This is for cortex area calculation
#                         #     polygon = Polygon(points)
#                         #     cortex_area += polygon.area
#                         cortex.append(points)
#                 elif (info['geometry']['type'] == 'MultiPolygon'):
#                     for ii in range(len(coordinates)):
#                         coor = coordinates[ii]
#                         coor = coor[0]
#                         points = []
#                         for jj in range(len(coor)):
#                             #            print(coordinates[j])
#                             points.append(coor[jj])
#
#                         top_left_x, top_left_y, bot_right_x, bot_right_y = bounding_box(points)
#                         if (((bot_right_x - top_left_x) * (bot_right_y - top_left_y)) > 30000):
#                             cortex_bbox.append([(top_left_x, top_left_y), (bot_right_x, bot_right_y)])
#                             # This is for cortex area calculation
#                             #     polygon = Polygon(points)
#                             #     cortex_area += polygon.area
#                             cortex.append(points)
#             if (prop_name == 'Positive'):
#                 print('This is the ifta annotation')
#                 coordinates = info['geometry']['coordinates']
#                 if (info['geometry']['type'] == 'Polygon'):
#                     points = []
#                     #         print(coordinates)
#                     coordinates = coordinates[0]
#                     for j in range(len(coordinates)):
#                         #            print(coordinates[j])
#                         points.append(coordinates[j])
#                     #        print(points)
#                     # return [(top_left_x, top_left_y), (bot_right_x, bot_right_y)]
#                     top_left_x, top_left_y, bot_right_x, bot_right_y = bounding_box(points)
#                     if (((bot_right_x - top_left_x) * (bot_right_y - top_left_y)) > 10000):
#                         # ifta_bbox.append([(top_left_x, top_left_y), (bot_right_x, bot_right_y)])
#                         # This is for cortex area calculation
#                         #     polygon = Polygon(points)
#                         #     cortex_area += polygon.area
#                         ifta.append(points)
#                 elif (info['geometry']['type'] == 'MultiPolygon'):
#                     for ii in range(len(coordinates)):
#                         coor = coordinates[ii]
#                         coor = coor[0]
#                         points = []
#                         for jj in range(len(coor)):
#                             #            print(coordinates[j])
#                             points.append(coor[jj])
#
#                         top_left_x, top_left_y, bot_right_x, bot_right_y = bounding_box(points)
#                         if (((bot_right_x - top_left_x) * (bot_right_y - top_left_y)) > 10000):
#                             # ifta_bbox.append([(top_left_x, top_left_y), (bot_right_x, bot_right_y)])
#                             # This is for cortex area calculation
#                             #     polygon = Polygon(points)
#                             #     cortex_area += polygon.area
#                             ifta.append(points)
#             if (prop_name == 'Other'):
#                 print('This is the pre-ifta annotation')
#                 coordinates = info['geometry']['coordinates']
#                 if (info['geometry']['type'] == 'Polygon'):
#                     points = []
#                     #         print(coordinates)
#                     coordinates = coordinates[0]
#                     for j in range(len(coordinates)):
#                         #            print(coordinates[j])
#                         points.append(coordinates[j])
#                     #        print(points)
#                     # return [(top_left_x, top_left_y), (bot_right_x, bot_right_y)]
#                     top_left_x, top_left_y, bot_right_x, bot_right_y = bounding_box(points)
#                     if (((bot_right_x - top_left_x) * (bot_right_y - top_left_y)) > 10000):
#                         # pre_ifta_bbox.append([(top_left_x, top_left_y), (bot_right_x, bot_right_y)])
#                         # This is for cortex area calculation
#                         #     polygon = Polygon(points)
#                         #     cortex_area += polygon.area
#                         pre_ifta.append(points)
#                 elif (info['geometry']['type'] == 'MultiPolygon'):
#                     for ii in range(len(coordinates)):
#                         coor = coordinates[ii]
#                         coor = coor[0]
#                         points = []
#                         for jj in range(len(coor)):
#                             #            print(coordinates[j])
#                             points.append(coor[jj])
#
#                         top_left_x, top_left_y, bot_right_x, bot_right_y = bounding_box(points)
#                         if (((bot_right_x - top_left_x) * (bot_right_y - top_left_y)) > 10000):
#                             # ifta_bbox.append([(top_left_x, top_left_y), (bot_right_x, bot_right_y)])
#                             # This is for cortex area calculation
#                             #     polygon = Polygon(points)
#                             #     cortex_area += polygon.area
#                             pre_ifta.append(points)
#
#         cortex_all.append(cortex)
#         cortex_bbox_all.append(cortex_bbox)
#         ifta_all.append(ifta)
#         pre_ifta_all.append(pre_ifta)
#
#         cortex_num.append(len(cortex))
#         cortex_bbox_num.append(len(cortex_bbox))
#         ifta_num.append(len(ifta))
#         pre_ifta_num.append(len(pre_ifta))
# #
# # print(cortex_bbox)
# # print(len(cortex))
#
# print(len(wsi_id_all))
# print(len(cortex_bbox_all))
# print(len(cortex_all))
# print(len(ifta_all))
# print(len(pre_ifta_all))
# print(len(cortex_num))
# print(len(cortex_bbox_num))
# print(len(ifta_num))
# print(len(pre_ifta_num))
# dataframe = pd.DataFrame({'wsi_id':wsi_id_all,'cortex_number':cortex_num,'cortex_bbox_num':cortex_bbox_num,'ifta_number':ifta_num,'pre_ifta_number':pre_ifta_num,'cortex_bbox':cortex_bbox_all,"cortex":cortex_all,'ifta':ifta_all,'pre_ifta':pre_ifta_all})
#
# dataframe.to_csv("./Neptune_points.csv",index=False,sep=',')
# #
