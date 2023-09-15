import unittest

import numpy as np
import tensorflow as tf

from dataset_functions import parse_dataset, x_y_split, ignore_weather_info

class TestIgnoreWeatherInfo(unittest.TestCase):

    def test_correct_handling(self):
        ds = parse_dataset('test/test.tfrecord', 'GZIP')
        xy_split = ds.map(x_y_split)
        xy_split_batch = xy_split.batch(4)
        for x in xy_split_batch:
            latent_vector_batch = x['latent_metadata']
            (
                input_platform_headings,
                input_incidence_angles,
                input_mission_ids,
                target_platform_heading,
                target_incidence_angle,
                target_mission_id,
            ) = ignore_weather_info(latent_vector_batch)

            for i, sample in enumerate(ds):
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

            target_data = tf.stack(
                [
                    target_platform_heading,
                    target_incidence_angle,
                    target_mission_id,
                ],
                axis=-1,
            )
            new_latent_vector = tf.concat(
                [
                    input_platform_headings,
                    input_incidence_angles,
                    input_mission_ids,
                    target_data,
                ],
                axis=-1
            )
            np.testing.assert_equal(new_latent_vector.shape[0], 4)

            for v, sample in zip(new_latent_vector, ds):
                np.testing.assert_array_equal(v[0:4].numpy(), sample['input_image_platform_headings'].numpy())
                np.testing.assert_array_equal(v[4:8].numpy(), sample['input_image_incidence_angles'].numpy())
                np.testing.assert_array_equal(v[8:12].numpy(), (sample['input_image_mission_ids'].numpy() == 'S1A'.encode('utf-8')).astype(np.float64))
                np.testing.assert_equal(v[12].numpy(), sample['target_image_platform_heading'].numpy())
                np.testing.assert_equal(v[13].numpy(), sample['target_image_incidence_angle'].numpy())
                np.testing.assert_equal(v[14].numpy(), float(sample['target_image_mission_id'].numpy() == 'S1A'.encode('utf-8')))

if __name__ == '__main__':
    unittest.main()
