import numpy as np
from functions import *
from shapely.geometry import Polygon
import cv2
import os
import json
import glob

json_files = glob.glob('/Volumes/data/UMich/thickened_TBM_test/cortex_json/*')
des_path = '/Volumes/data/UMich/thickened_TBM_test/cortex_image/'
for json_file in json_files:
    fileID = os.path.basename(json_file).split('.')[0]
    wsi_file = glob.glob(f'/Volumes/data/UMich/thickened_TBM_test/WSI/{fileID}*')[0]
    slide = openslide.OpenSlide(wsi_file)
    with open(json_file, 'r') as load_f:
       load_dict = json.load(load_f)

    cortexes = get_structure(load_dict,classname=['cortex_QCed'])
    print(len(cortexes))
    for cortex in cortexes:
        if cortex.area > 1000000:
            minx, miny, maxx, maxy = cortex.bounds
            minx, miny, maxx, maxy = minx-50, miny-50, maxx+50, maxy+50

            pas = cv2.cvtColor(np.asarray(slide.read_region((int(minx), int(miny)), 0, (int((maxx - minx)), int((maxy - miny)))))[:, :,0:3], cv2.COLOR_RGB2BGR)
            pas = cv2.resize(pas, (0, 0), fx = 0.25, fy = 0.25)
            cv2.imwrite(f'/Volumes/data/UMich/thickened_TBM_test/cortex_image/{fileID}_{minx}_{miny}.png',pas)