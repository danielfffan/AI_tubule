#%%
#%%

import os
import cv2
import argparse
import numpy as np
import json
from skimage import color
import openslide
import pandas as pd
import glob
import csv
import shutil


#%%
with open('/Volumes/data/Neptune/AI_tubule_biopsyid.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    column = [row for row in reader]
wsi_folders = '/Volumes/data/Neptune/FSGS_MCD/'

#%%


#Get the mapping csv file
#check if the cortex bounding box is correct
#map the tubule ROI from 10X to 40X
safids = glob.glob('tubule_tbm/*')
resize = 0.25
for safid in safids:
    print(safid)
    geojson_features = []
    bbox_images = glob.glob(f'{safid}/*/')
    for bbox_image in bbox_images:
        print(bbox_image)
        # print(bbox_image.split('_'))
        x_roi,y_roi,w_roi,h_roi = float(bbox_image.split('_')[3]),float(bbox_image.split('_')[5]),float(bbox_image.split('_')[7]),float(bbox_image.split('_')[9])
        # print(x_roi,y_roi,w_roi,h_roi)
        wsiid = bbox_image.split('_')[1]
        # print(wsiid)
        tubule_images = glob.glob(f'{bbox_image}/single_tubule/tubule/*')
        for tubule_image in tubule_images:
            index = os.path.basename(tubule_image).split('[')[1].split(',')
            print(index)
            x_tile = int(float(index[0].split('x=')[1]))
            y_tile = int(float(index[1].split('y=')[1]))
            print(x_tile,y_tile)
            binary_contour = cv2.imread(tubule_image, cv2.IMREAD_GRAYSCALE)
            binary_mask = cv2.threshold(binary_contour, 127, 255, cv2.THRESH_BINARY)[1]
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_coords = np.squeeze(contour).tolist()
                contour_coords.append(contour_coords[0])


                contour_coords = [[4*(c[0]+x_tile)+x_roi, 4*(c[1]+y_tile)+y_roi] for c in contour_coords]
                contour_coords.append(contour_coords[0])
                # Create GeoJSON feature for image contour
                geojson_feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [contour_coords]
                    },
                     "properties": {
                          "objectType": "annotation",
                          "classification": {
                            "name": "Tubule",
                            "color": [96, 12, 26]
                          }
                        }
                }

                # Append GeoJSON feature to list
                geojson_features.append(geojson_feature)

    # Create GeoJSON object with all features
    geojson_obj = {
    "type": "FeatureCollection",
    "features": geojson_features
}

# Save combined GeoJSON object to file
    with open(f'{safid}/tubule.geojson','w') as f:
        json.dump(geojson_obj, f)

