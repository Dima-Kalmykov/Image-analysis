from osgeo import gdal
import numpy as np
import os

import geopandas as gpd
import pandas as pd


def save_to_shp_file(points, path):
    start_part = "C:/temp/landsat/"
    points.to_file(start_part + path)


file_with_points = gpd.read_file('C:/Users/dmkal/Documents/truth_data.shp')
class_names = file_with_points['lctype'].unique()

print('class names:', class_names)
class_ids = np.arange(class_names.size) + 1

print('class ids:', class_ids)
print("\nPoints data without id:\n")
print(file_with_points.head())
file_with_points['id'] = file_with_points['lctype'].map(dict(zip(class_names, class_ids)))

print("\nPoint data with id:\n")
print(file_with_points.head())

train_points = file_with_points.sample(frac=0.7)

# Difference between all points and train points
test_points = file_with_points.drop(train_points.index)
print('\ntotal shape', file_with_points.shape,
      'training shape', train_points.shape,
      'test', test_points.shape)

save_to_shp_file(train_points, "train.shp")
save_to_shp_file(test_points, "test.shp")
