import csv
import csv

with open('./Neptune.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    column = [row for row in reader]
import glob

files = []
wsiids = []
for i in range(len(column)):
    p_information = column[i]
    filename = p_information['filename'
    ]
    # print(filename)
    files_ = glob.glob(f'/Volumes/data/Neptune/FSGS_MCD/{filename}')
    if (len(files_) != 0):
        files.append(filename)
        wsiids.append(p_information['WSI_ID'])
print(len(files),len(wsiids))

import numpy as np
from crop_roi_function import *
from shapely.geometry import Polygon
import cv2
import math
import os
import staintools
import csv
import openslide
import glob
import skimage.io as skio
from ast import literal_eval
csv.field_size_limit(sys.maxsize)

ite_num = 'image'
resize = 0.25
print(files)
for j in range(len(files)):
    wsi_index = wsiids[j]
    if ('.svs' in files[j]):
        wsi_id = files[j].split('.svs')[0]
    elif('.ndpi' in files[j]):
        wsi_id = files[j].split('.ndpi')[0]

    wsi_files = glob.glob(f'/Volumes/data/Neptune/FSGS_MCD/{wsi_id}*')
    wsi_filename = wsi_files[0]
    print(wsi_filename)
    des_folder = f'{ite_num}/{wsi_index}'
    #os.mkdir(des_folder)
    slide = openslide.OpenSlide(wsi_filename)
    flag_name = os.path.exists(des_folder)

    with open('Neptune_points.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        column = [row for row in reader]

        #Crop the cortex tissue img
    roi_file_names = []
    for i in range(len(column)):
        info = column[i]
        wsi_id_reg = info['wsi_id']
        # print(wsi_id_reg)
        if (wsi_id_reg == wsi_id):
            cortex_bbox = info['cortex_bbox']
            cortex_bbox = list(literal_eval(cortex_bbox))
            cortex = info['cortex']
            cortex = list(literal_eval(cortex))
            ifta = info['ifta']
            ifta = list(literal_eval(ifta))
            pre_ifta = info['pre_ifta']
            pre_ifta = list(literal_eval(pre_ifta))
            print(f'###########{wsi_id}############')
            # Crop the cortex mask
            print(f"There are {len(cortex)} cortex areas!")
            flag_cortex = 0
            for cor in cortex_bbox:
                print(f"This is for the {flag_cortex}th bounding box.")
                x_min = cor[0][0] - 200
                x_max = cor[1][0] + 200
                y_min = cor[0][1] - 200
                y_max = cor[1][1] + 200
                # print(x_min,x_max,y_min,y_max)
                x_dis = int(x_max - x_min)
                y_dis = int(y_max - y_min)
                roi_area = x_dis * y_dis
                x_min = int(x_min)
                y_min = int(y_min)
                print(f"The original bounding box area is {roi_area}")
                #       print(f"The original bounding box area is 100000000")
                tissue_roi_name = f"{des_folder}/{wsi_index}_x_{x_min}_y_{y_min}_w_{x_dis}_h_{y_dis}_{flag_cortex}_tissue.png"

                print("Checking if file already exists...")
                flag_name = os.path.exists(tissue_roi_name)
                if (flag_name == False):
                    print('File does not exist! Storing......')
                    roi = np.asarray(slide.read_region((int(x_min), int(y_min)), 0, (int((x_dis)), int((y_dis)))))[:, :,
                          0:3]
                    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                    roi = cv2.resize(roi, (0, 0), fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(tissue_roi_name, roi)
                    roi_file_names.append(tissue_roi_name)
                    print("Done!")
                else:
                    roi_file_names.append(tissue_roi_name)
                    print("file already exists!")
                flag_cortex += 1
            # remove background
            # print(len(cortex))
            print('Now for removing the background')
            # print(roi_file_names)
            for roi_file_name in roi_file_names:
                print(roi_file_name)
                filename = os.path.basename(roi_file_name)
                # get x_min
                x_min = int(int(filename.split('x_')[1].split('_')[0]) * resize)
                y_min = int(int(filename.split('y_')[1].split('_')[0]) * resize)
                print(x_min, y_min)
                index = int(filename.split('h_')[1].split('_')[1])
                print(index)
                points = cortex[index]
                points_copy = []
                for i in range(len(points)):
                    point_copy = [int(points[i][0] * resize), int(points[i][1] * resize)]
                    points_copy.append(point_copy)
                roi_rmbg_name = roi_file_name.replace('tissue', 'tissue_rmbg')
                roi_cor_mask = roi_file_name.replace('tissue', 'cortex_mask')
                print("Doing the remove background process......")
                if (os.path.exists(roi_rmbg_name) == False):
                    roi = cv2.imread(roi_file_name)
                    if (get_tissue_mask(roi, luminosity_threshold=0.2)):
                        print("Removing the background......")
                        points_ = []
                        for jj in range(len(points_copy)):
                            point = points_copy[jj]
                            point[0] = int((point[0] - x_min))
                            point[1] = int((point[1] - y_min))
                            points_.append(point)
                        pts = np.array(points_)
                        croped = roi.copy()
                        mask = np.zeros(croped.shape[:2], np.uint8)
                        mask = cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.imwrite(roi_cor_mask, mask)
                        roi = cv2.imread(roi_file_name)
                        croped = roi.copy()
                        dst = cv2.bitwise_and(croped, croped, mask=mask)
                        bg = np.ones_like(croped, np.uint8)
                        r, g, b = rgb_median(roi, mask)
                        bg[:, :, 0] = bg[:, :, 0] * r
                        bg[:, :, 1] = bg[:, :, 1] * g
                        bg[:, :, 2] = bg[:, :, 2] * b
                        bg = cv2.bitwise_and(bg, bg, mask=mask)
                        dst2 = bg + dst
                        cv2.imwrite(roi_rmbg_name, dst2)
                        print('done!')
                else:
                    print('File already exists!')
