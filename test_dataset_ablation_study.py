import unittest

import numpy as np
import tensorflow as tf

from dataset_functions import parse_dataset, x_y_split, include_condition_data

class TestAblationStudy(unittest.TestCase):

    def test_drop_temperature_data(self):
        ds = parse_dataset('test/test.tfrecord', 'GZIP')
        xy_split = ds.map(x_y_split)
        xy_split_batch = xy_split.batch(4)
        for x in xy_split_batch:
            latent_vector_batch = x['latent_metadata']
            (
                input_snow_depths,
                input_platform_headings,
                input_incidence_angles,
                input_precipitation,
                input_mission_ids,
                target_snow_depth,
                target_platform_heading,
                target_incidence_angle,
                target_mission_id,
                target_precipitation,
            ) = include_condition_data(latent_vector_batch, temperature=False)

            for i, sample in enumerate(ds):
                np.testing.assert_array_equal(
                    sample['input_image_snow_depths'].numpy(),
                    input_snow_depths[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_platform_headings'].numpy(),
                    input_platform_headings[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_incidence_angles'].numpy(),
                    input_incidence_angles[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_precipitations'].numpy().flatten(),
                    input_precipitation[i].numpy(),
                )
                np.testing.assert_array_equal(
                    (sample['input_image_mission_ids'].numpy() == 'S1A'.encode('utf-8')).astype(np.float64),
                    input_mission_ids[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_snow_depth'].numpy(),
                    target_snow_depth[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_platform_heading'].numpy(),
                    target_platform_heading[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_incidence_angle'].numpy(),
                    target_incidence_angle[i].numpy(),
                )
                np.testing.assert_equal(
                    float(sample['target_image_mission_id'].numpy() == 'S1A'.encode('utf-8')),
                    target_mission_id[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['target_image_precipitations'].numpy(),
                    target_precipitation[i].numpy(),
                )


    def test_drop_snow_depth_data(self):
        ds = parse_dataset('test/test.tfrecord', 'GZIP')
        xy_split = ds.map(x_y_split)
        xy_split_batch = xy_split.batch(4)
        for x in xy_split_batch:
            latent_vector_batch = x['latent_metadata']
            (
                input_temperatures,
                input_platform_headings,
                input_incidence_angles,
                input_precipitation,
                input_mission_ids,
                target_temperature,
                target_platform_heading,
                target_incidence_angle,
                target_mission_id,
                target_precipitation,
            ) = include_condition_data(latent_vector_batch, snow_depth=False)

            for i, sample in enumerate(ds):
                np.testing.assert_array_equal(
                    sample['input_image_temperatures'].numpy(),
                    input_temperatures[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_platform_headings'].numpy(),
                    input_platform_headings[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_incidence_angles'].numpy(),
                    input_incidence_angles[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_precipitations'].numpy().flatten(),
                    input_precipitation[i].numpy(),
                )
                np.testing.assert_array_equal(
                    (sample['input_image_mission_ids'].numpy() == 'S1A'.encode('utf-8')).astype(np.float64),
                    input_mission_ids[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_temperature'].numpy(),
                    target_temperature[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_platform_heading'].numpy(),
                    target_platform_heading[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_incidence_angle'].numpy(),
                    target_incidence_angle[i].numpy(),
                )
                np.testing.assert_equal(
                    float(sample['target_image_mission_id'].numpy() == 'S1A'.encode('utf-8')),
                    target_mission_id[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['target_image_precipitations'].numpy(),
                    target_precipitation[i].numpy(),
                )


    def test_drop_platform_heading_data(self):
        ds = parse_dataset('test/test.tfrecord', 'GZIP')
        xy_split = ds.map(x_y_split)
        xy_split_batch = xy_split.batch(4)
        for x in xy_split_batch:
            latent_vector_batch = x['latent_metadata']
            (
                input_temperatures,
                input_snow_depths,
                input_incidence_angles,
                input_precipitation,
                input_mission_ids,
                target_temperature,
                target_snow_depth,
                target_incidence_angle,
                target_mission_id,
                target_precipitation,
            ) = include_condition_data(latent_vector_batch, platform_heading=False)

            for i, sample in enumerate(ds):
                np.testing.assert_array_equal(
                    sample['input_image_temperatures'].numpy(),
                    input_temperatures[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_snow_depths'].numpy(),
                    input_snow_depths[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_incidence_angles'].numpy(),
                    input_incidence_angles[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_precipitations'].numpy().flatten(),
                    input_precipitation[i].numpy(),
                )
                np.testing.assert_array_equal(
                    (sample['input_image_mission_ids'].numpy() == 'S1A'.encode('utf-8')).astype(np.float64),
                    input_mission_ids[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_temperature'].numpy(),
                    target_temperature[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_snow_depth'].numpy(),
                    target_snow_depth[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_incidence_angle'].numpy(),
                    target_incidence_angle[i].numpy(),
                )
                np.testing.assert_equal(
                    float(sample['target_image_mission_id'].numpy() == 'S1A'.encode('utf-8')),
                    target_mission_id[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['target_image_precipitations'].numpy(),
                    target_precipitation[i].numpy(),
                )


    def test_drop_incidence_angle(self):
        ds = parse_dataset('test/test.tfrecord', 'GZIP')
        xy_split = ds.map(x_y_split)
        xy_split_batch = xy_split.batch(4)
        for x in xy_split_batch:
            latent_vector_batch = x['latent_metadata']
            (
                input_temperatures,
                input_snow_depths,
                input_platform_headings,
                input_precipitation,
                input_mission_ids,
                target_temperature,
                target_snow_depth,
                target_platform_heading,
                target_mission_id,
                target_precipitation,
            ) = include_condition_data(latent_vector_batch, incidence_angle=False)

            for i, sample in enumerate(ds):
                np.testing.assert_array_equal(
                    sample['input_image_temperatures'].numpy(),
                    input_temperatures[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_snow_depths'].numpy(),
                    input_snow_depths[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_platform_headings'].numpy(),
                    input_platform_headings[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_precipitations'].numpy().flatten(),
                    input_precipitation[i].numpy(),
                )
                np.testing.assert_array_equal(
                    (sample['input_image_mission_ids'].numpy() == 'S1A'.encode('utf-8')).astype(np.float64),
                    input_mission_ids[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_temperature'].numpy(),
                    target_temperature[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_snow_depth'].numpy(),
                    target_snow_depth[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_platform_heading'].numpy(),
                    target_platform_heading[i].numpy(),
                )
                np.testing.assert_equal(
                    float(sample['target_image_mission_id'].numpy() == 'S1A'.encode('utf-8')),
                    target_mission_id[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['target_image_precipitations'].numpy(),
                    target_precipitation[i].numpy(),
                )


    def test_drop_temperature_data(self):
        ds = parse_dataset('test/test.tfrecord', 'GZIP')
        xy_split = ds.map(x_y_split)
        xy_split_batch = xy_split.batch(4)
        for x in xy_split_batch:
            latent_vector_batch = x['latent_metadata']
            (
                input_temperatures,
                input_snow_depths,
                input_platform_headings,
                input_incidence_angles,
                input_mission_ids,
                target_temperature,
                target_snow_depth,
                target_platform_heading,
                target_incidence_angle,
                target_mission_id,
            ) = include_condition_data(latent_vector_batch, precipitation=False)

            for i, sample in enumerate(ds):
                np.testing.assert_array_equal(
                    sample['input_image_temperatures'].numpy(),
                    input_temperatures[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_snow_depths'].numpy(),
                    input_snow_depths[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_platform_headings'].numpy(),
                    input_platform_headings[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_incidence_angles'].numpy(),
                    input_incidence_angles[i].numpy(),
                )
                np.testing.assert_array_equal(
                    (sample['input_image_mission_ids'].numpy() == 'S1A'.encode('utf-8')).astype(np.float64),
                    input_mission_ids[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_temperature'].numpy(),
                    target_temperature[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_snow_depth'].numpy(),
                    target_snow_depth[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_platform_heading'].numpy(),
                    target_platform_heading[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_incidence_angle'].numpy(),
                    target_incidence_angle[i].numpy(),
                )
                np.testing.assert_equal(
                    float(sample['target_image_mission_id'].numpy() == 'S1A'.encode('utf-8')),
                    target_mission_id[i].numpy(),
                )


    def test_drop_mission_ids(self):
        ds = parse_dataset('test/test.tfrecord', 'GZIP')
        xy_split = ds.map(x_y_split)
        xy_split_batch = xy_split.batch(4)
        for x in xy_split_batch:
            latent_vector_batch = x['latent_metadata']
            (
                input_temperatures,
                input_snow_depths,
                input_platform_headings,
                input_incidence_angles,
                input_precipitation,
                target_temperature,
                target_snow_depth,
                target_platform_heading,
                target_incidence_angle,
                target_precipitation,
            ) = include_condition_data(latent_vector_batch, mission_ids=False)

            for i, sample in enumerate(ds):
                np.testing.assert_array_equal(
                    sample['input_image_temperatures'].numpy(),
                    input_temperatures[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_snow_depths'].numpy(),
                    input_snow_depths[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_platform_headings'].numpy(),
                    input_platform_headings[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_incidence_angles'].numpy(),
                    input_incidence_angles[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['input_image_precipitations'].numpy().flatten(),
                    input_precipitation[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_temperature'].numpy(),
                    target_temperature[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_snow_depth'].numpy(),
                    target_snow_depth[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_platform_heading'].numpy(),
                    target_platform_heading[i].numpy(),
                )
                np.testing.assert_equal(
                    sample['target_image_incidence_angle'].numpy(),
                    target_incidence_angle[i].numpy(),
                )
                np.testing.assert_array_equal(
                    sample['target_image_precipitations'].numpy(),
                    target_precipitation[i].numpy(),
                )


if __name__ == '__main__':
    unittest.main()
