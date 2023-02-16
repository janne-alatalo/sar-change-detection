import random
import json

import numpy as np
from osgeo import ogr, osr

import db_functions
from shape_generator import get_random_shape_raster
import statsmodels.api as sm
from scipy.interpolate import interp1d


def compute_superpixels(img):
    height, width = img.shape[0:2]

    superpixel_sets = {}
    superpixel_map = np.full((height, width), -1, dtype=np.int64)

    autoincrement_id = 1

    for y in range(height):
        for x in range(width):
            pixel_value = img[y, x]
            # is part of upper superpixel
            if y != 0 and np.all(pixel_value == img[y - 1, x]):
                upper_superpixel_id = superpixel_map[y - 1, x]
                superpixel_sets[upper_superpixel_id].append((x, y))
                superpixel_map[y, x] = upper_superpixel_id
            # is part of left superpixel
            elif x != 0 and np.all(pixel_value == img[y, x - 1]):
                left_superpixel_id = superpixel_map[y, x - 1]
                superpixel_sets[left_superpixel_id].append((x, y))
                superpixel_map[y, x] = left_superpixel_id
            # no part of an existing superpixel
            else:
                new_superpixel_set = [(x, y)]
                new_superpixel_set_id = autoincrement_id
                autoincrement_id += 1
                superpixel_sets[new_superpixel_set_id] = new_superpixel_set
                superpixel_map[y, x] = new_superpixel_set_id

    return superpixel_map, superpixel_sets


def get_simulated_change_maps(cursor, img_arr, srid3067_geotransform, num_changes=(0, 5)):

    y_size, x_size = img_arr.shape[0:2]
    # Hacky fix for the problem where the db query throws "Self-intersection error"
    # Use full forestmask in that case
    try:
        swamp_forestmask_arr = db_functions.get_non_forest_area_geometry_arr(cursor, srid3067_geotransform, (x_size, y_size))
    except:
        swamp_forestmask_arr = np.full((y_size, x_size), 0.0, dtype=np.float32)

    num_changes = random.randint(*num_changes)
    all_random_shapes = []
    all_change_coords_centered = []
    all_change_coords = []

    # It might be possible that the image doesn't have any forest areas
    if not np.all(swamp_forestmask_arr == 0.0):
        for _ in range(num_changes):
            while True:
                rand_x = random.randint(0, x_size - 1)
                rand_y = random.randint(0, y_size - 1)
                if swamp_forestmask_arr[rand_y, rand_x] == 1.0:
                    break
            random_shape = get_random_shape_raster(100, 100)

            s = random_shape.shape
            centered_x = rand_x - int(np.round(s[1] / 2))
            centered_y = rand_y - int(np.round(s[0] / 2))

            all_random_shapes.append(random_shape)
            all_change_coords_centered.append((centered_x, centered_y))
            all_change_coords.append((rand_x, rand_y))

    target_img_superpixel_map, target_img_superpixel_sets = compute_superpixels(img_arr)
    combined_change_mask_arr = np.full(img_arr.shape[0:2], 0, dtype=np.int32)

    for i, (random_shape, change_coords) in enumerate(zip(all_random_shapes, all_change_coords_centered)):
        (centered_x, centered_y) = change_coords
        s = random_shape.shape

        change_mask = np.zeros_like(swamp_forestmask_arr)
        random_shape_cut_low_x = max(0, centered_x * -1)
        random_shape_cut_high_x = s[1] + min(change_mask.shape[1] - (centered_x + s[1]), 0)
        random_shape_cut_low_y = max(0, centered_y * -1)
        random_shape_cut_high_y = s[0] + min(change_mask.shape[0] - (centered_y + s[0]), 0)

        random_shape_cut_x = slice(random_shape_cut_low_x, random_shape_cut_high_x)
        random_shape_cut_y = slice(random_shape_cut_low_y, random_shape_cut_high_y)

        change_mask_slice_x = slice(max(0, centered_x), centered_x + s[1])
        change_mask_slice_y = slice(max(0, centered_y), centered_y + s[0])
        change_mask[change_mask_slice_y, change_mask_slice_x] = random_shape[random_shape_cut_y, random_shape_cut_x]
        combined_change_mask_arr[(change_mask == True) & (swamp_forestmask_arr == 1.0)] = i + 1

        target_img_superpixe_map_copy = np.copy(target_img_superpixel_map)
        target_img_superpixe_map_copy[(change_mask == False) | (swamp_forestmask_arr == 0.0)] = -1

    return (
        num_changes,
        combined_change_mask_arr,
        target_img_superpixel_map,
        target_img_superpixel_sets,
        swamp_forestmask_arr,
    )


def add_random_noise(img, num_changes, change_mask_arr, img_superpixel_map, img_superpixel_sets, change_loc, change_scale):
    img_copy = np.copy(img)
    for i in range(1, num_changes + 1):
        img_superpixel_map_copy = np.copy(img_superpixel_map)
        img_superpixel_map_copy[change_mask_arr != i] = -1
        all_superpixel_ids = np.unique(img_superpixel_map_copy.flatten())

        for superpixel_id in all_superpixel_ids:
            if superpixel_id == -1:
                continue
            random_noise = np.random.normal(loc=change_loc, scale=change_scale, size=2)
            for x, y in img_superpixel_sets[superpixel_id]:
                img_copy[y, x] += random_noise

    return img_copy


def add_statistical_change(
        cursor,
        image_location,
        data_take_id,
        start_time,
        image,
        num_changes,
        area_size_meters,
    ):

    img_arr = image.transpose([1, 2, 0])
    img_copy = np.copy(img_arr)

    target_img_superpixel_map, target_img_superpixel_sets = compute_superpixels(img_arr)

    wgs80_srs = osr.SpatialReference()
    wgs80_srs.ImportFromEPSG(4326)
    fin_srs = osr.SpatialReference()
    fin_srs.ImportFromEPSG(3067)

    point_epsg4326 = ogr.CreateGeometryFromWkt(image_location)
    point_epsg4326.AssignSpatialReference(wgs80_srs)
    position_epsg4326 = point_epsg4326.GetY(), point_epsg4326.GetX()

    point_epsg3067 = point_epsg4326.Clone()
    point_epsg3067.FlattenTo2D()
    point_epsg3067.TransformTo(fin_srs)

    muuavoinalue_arrays = db_functions.get_n_closest_maastotietokanta_muuavoinalue_from_raster(
        cursor,
        position_epsg4326,
        data_take_id,
        start_time,
        50,
    )
    if muuavoinalue_arrays is None:
        print('Image not found')
        return None

    muuavoinalue_band1_pixels = np.concatenate([a[0, :, :].flatten() for a in muuavoinalue_arrays])
    muuavoinalue_band1_pixels = muuavoinalue_band1_pixels[~np.isnan(muuavoinalue_band1_pixels)]
    muuavoinalue_band2_pixels = np.concatenate([a[1, :, :].flatten() for a in muuavoinalue_arrays])
    muuavoinalue_band2_pixels = muuavoinalue_band2_pixels[~np.isnan(muuavoinalue_band2_pixels)]

    _, y_size, x_size = image.shape
    offset = area_size_meters / 2

    geotransform = [
        point_epsg3067.GetX() - offset,
        area_size_meters / x_size,
        0,
        point_epsg3067.GetY() + offset,
        0,
        -area_size_meters / y_size,
    ]
    non_forest_mask = db_functions.get_maastotietokanta_non_forest_mask_arr(cursor, geotransform, (x_size, y_size))

    forest_mask = np.zeros_like(non_forest_mask)
    forest_mask[non_forest_mask == 0.0] = 1.0

    band1 = img_arr[:, :, 0]
    band2 = img_arr[:, :, 1]

    forest_data_band1 = band1[forest_mask == 1.0].flatten()
    forest_data_band1 = forest_data_band1[~np.isnan(forest_data_band1)]

    forest_data_band2 = band2[forest_mask == 1.0].flatten()
    forest_data_band2 = forest_data_band2[~np.isnan(forest_data_band2)]

    gridsize = 5000

    band1_model1 = sm.nonparametric.KDEUnivariate(forest_data_band1)
    band1_model1.fit(gridsize=gridsize)

    band1_model2 = sm.nonparametric.KDEUnivariate(muuavoinalue_band1_pixels)
    band1_model2.fit(gridsize=gridsize)

    map_band1 = interp1d(band1_model1.icdf[3:-3], band1_model2.icdf[3:-3], fill_value='extrapolate')

    band2_model1 = sm.nonparametric.KDEUnivariate(forest_data_band2)
    band2_model1.fit(gridsize=gridsize)

    band2_model2 = sm.nonparametric.KDEUnivariate(muuavoinalue_band2_pixels)
    band2_model2.fit(gridsize=gridsize)

    map_band2 = interp1d(band2_model1.icdf[3:-3], band2_model2.icdf[3:-3], fill_value='extrapolate')

    num_changes = random.randint(*num_changes)
    all_random_shapes = []
    all_change_coords_centered = []
    all_change_coords = []

    # It might be possible that the image doesn't have any forest areas
    if not np.all(forest_mask == 0.0):
        for _ in range(num_changes):
            while True:
                rand_x = random.randint(0, x_size - 1)
                rand_y = random.randint(0, y_size - 1)
                if forest_mask[rand_y, rand_x] == 1.0:
                    break
            random_shape = get_random_shape_raster(100, 100)

            s = random_shape.shape
            centered_x = rand_x - int(np.round(s[1] / 2))
            centered_y = rand_y - int(np.round(s[0] / 2))

            all_random_shapes.append(random_shape)
            all_change_coords_centered.append((centered_x, centered_y))
            all_change_coords.append((rand_x, rand_y))
    else:
        num_changes = 0

    combined_change_mask_arr = np.full(img_arr.shape[0:2], 0, dtype=np.int32)

    for i, (random_shape, change_coords) in enumerate(zip(all_random_shapes, all_change_coords_centered)):
        (centered_x, centered_y) = change_coords
        s = random_shape.shape

        change_mask = np.zeros_like(forest_mask)
        random_shape_cut_low_x = max(0, centered_x * -1)
        random_shape_cut_high_x = s[1] + min(change_mask.shape[1] - (centered_x + s[1]), 0)
        random_shape_cut_low_y = max(0, centered_y * -1)
        random_shape_cut_high_y = s[0] + min(change_mask.shape[0] - (centered_y + s[0]), 0)

        random_shape_cut_x = slice(random_shape_cut_low_x, random_shape_cut_high_x)
        random_shape_cut_y = slice(random_shape_cut_low_y, random_shape_cut_high_y)

        change_mask_slice_x = slice(max(0, centered_x), centered_x + s[1])
        change_mask_slice_y = slice(max(0, centered_y), centered_y + s[0])
        change_mask[change_mask_slice_y, change_mask_slice_x] = random_shape[random_shape_cut_y, random_shape_cut_x]
        combined_change_mask_arr[(change_mask == True) & (forest_mask == 1.0)] = i + 1

        target_img_superpixe_map_copy = np.copy(target_img_superpixel_map)
        target_img_superpixe_map_copy[(change_mask == False) | (forest_mask == 0.0)] = -1

    for i in range(1, num_changes + 1):
        img_superpixel_map_copy = np.copy(target_img_superpixel_map)
        img_superpixel_map_copy[combined_change_mask_arr != i] = -1
        all_superpixel_ids = np.unique(img_superpixel_map_copy.flatten())

        for superpixel_id in all_superpixel_ids:
            if superpixel_id == -1:
                continue
            band1_val = None
            band2_val = None
            for x, y in target_img_superpixel_sets[superpixel_id]:
                if band1_val is None and band2_val is None:
                    band1_val = map_band1(img_arr[y, x, 0])
                    band2_val = map_band2(img_arr[y, x, 1])
                img_copy[y, x] = [band1_val, band2_val]

    return (
        img_copy,
        num_changes,
        combined_change_mask_arr,
        target_img_superpixel_map,
        target_img_superpixel_sets,
        forest_mask,
    )
