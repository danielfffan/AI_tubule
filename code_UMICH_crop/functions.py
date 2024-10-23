import cv2
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import csv
import numpy as np
import geopandas
import os
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
import glob
import json
import openslide
from tqdm import tqdm

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage import io, color, filters, morphology, segmentation,exposure
from skimage.color import label2rgb

def get_structure(load_dict,classname):
    structure = []
    if len(classname) == 1:
        for load_dict_i in tqdm(load_dict, desc=f"Extracting {classname}"):
            if 'classification' in  load_dict_i['properties']:
                prop_name = load_dict_i['properties']['classification']['name']
                if (prop_name == classname[0]):
                    contours = load_dict_i['geometry']['coordinates']
                    if (load_dict_i['geometry']['type'] == 'MultiPolygon'):
                        for contour in contours:
                            points = [pt for pt in contour[0]]
                            structure.append(Polygon(points))
                    elif (load_dict_i['geometry']['type'] == 'Polygon'):
                        for contour in contours:
                            points = [pt for pt in contour]
                            structure.append(Polygon(points))
    elif(len(classname)==2):
        for load_dict_i in tqdm(load_dict, desc=f"Extracting {classname}"):
            if 'classification' in  load_dict_i['properties']:
                prop_name = load_dict_i['properties']['classification']['name']
                if (prop_name == classname[0] or prop_name==classname[1]):
                    contours = load_dict_i['geometry']['coordinates']
                    if (load_dict_i['geometry']['type'] == 'MultiPolygon'):
                        for contour in contours:
                            points = [pt for pt in contour[0]]
                            structure.append(Polygon(points))
                    elif (load_dict_i['geometry']['type'] == 'Polygon'):
                        for contour in contours:
                            points = [pt for pt in contour]
                            structure.append(Polygon(points))
    # result = structure[0]
    # for polygon in tqdm(structure[1:],desc=f'Post processing the {classname},N=={len(structure)}'):
    #     result = result.union(polygon)
    return structure