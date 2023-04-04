import tensorflow as tf
import numpy as np

# code from here https://www.tensorflow.org/tutorials/load_data/tfrecord
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _serialized_bytes_feature(value):
    return _bytes_feature(tf.io.serialize_tensor(value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_simulated_change_example(
    image_location_wkt,
    dem_rast,
    forestmask,
    target_image,
    target_image_start_time,
    target_image_data_take_id,
    target_image_incidence_angle,
    target_image_platform_heading,
    target_image_temperature,
    target_image_precipitations,
    target_image_snow_depth,
    target_image_mission_id,
    input_image_stack,
    input_image_start_times,
    input_image_data_take_ids,
    input_image_incidence_angles,
    input_image_platform_headings,
    input_image_temperatures,
    input_image_precipitations,
    input_image_snow_depths,
    input_image_mission_ids,
    num_simulated_changes,
    simulated_change_mask,
    img_superpixel_map,
    img_superpixel_sets_json,
    simulated_change_image,
    target_image_prediction=None,
):
    feature = {
        'image_location': _bytes_feature(image_location_wkt.encode('utf-8')),
        'dem_rast': _serialized_bytes_feature(dem_rast),
        'forestmask': _serialized_bytes_feature(forestmask),

        'target_image': _serialized_bytes_feature(target_image),
        'target_image_temperature': _float_feature(target_image_temperature),
        'target_image_snow_depth': _float_feature(target_image_snow_depth),
        'target_image_precipitations': _serialized_bytes_feature(target_image_precipitations),
        'target_image_start_time': _bytes_feature(target_image_start_time.encode('utf-8')),
        'target_image_data_take_id': _bytes_feature(target_image_data_take_id.encode('utf-8')),
        'target_image_incidence_angle': _float_feature(target_image_incidence_angle),
        'target_image_platform_heading': _float_feature(target_image_platform_heading),
        'target_image_mission_id': _bytes_feature(target_image_mission_id.encode('utf-8')),

        'input_image_stack': _serialized_bytes_feature(input_image_stack),
        'input_image_temperatures': _serialized_bytes_feature(input_image_temperatures),
        'input_image_precipitations': _serialized_bytes_feature(input_image_precipitations),
        'input_image_snow_depths': _serialized_bytes_feature(input_image_snow_depths),
        'input_image_start_times': _serialized_bytes_feature(input_image_start_times),
        'input_image_data_take_ids': _serialized_bytes_feature(input_image_data_take_ids),
        'input_image_incidence_angles': _serialized_bytes_feature(input_image_incidence_angles),
        'input_image_platform_headings': _serialized_bytes_feature(input_image_platform_headings),
        'input_image_mission_ids': _serialized_bytes_feature(input_image_mission_ids),
        'simulated_change_mask': _serialized_bytes_feature(simulated_change_mask),
        'num_simulated_changes': _int64_feature(num_simulated_changes),
        'img_superpixel_map': _serialized_bytes_feature(img_superpixel_map),
        'img_superpixel_sets': _bytes_feature(img_superpixel_sets_json.encode('utf-8')),

        'simulated_change_image': _serialized_bytes_feature(simulated_change_image),
    }

    if target_image_prediction is not None:
        feature['target_image_prediction'] = _serialized_bytes_feature(target_image_prediction)

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def serialize_example(
    image_location_wkt,
    dem_rast,
    forestmask,
    target_image,
    target_image_start_time,
    target_image_data_take_id,
    target_image_incidence_angle,
    target_image_platform_heading,
    target_image_temperature,
    target_image_precipitations,
    target_image_snow_depth,
    target_image_mission_id,
    input_image_stack,
    input_image_start_times,
    input_image_data_take_ids,
    input_image_incidence_angles,
    input_image_platform_headings,
    input_image_temperatures,
    input_image_precipitations,
    input_image_snow_depths,
    input_image_mission_ids,
):
    feature = {
        'image_location': _bytes_feature(image_location_wkt.encode('utf-8')),
        'dem_rast': _serialized_bytes_feature(dem_rast),
        'forestmask': _serialized_bytes_feature(forestmask),

        'target_image': _serialized_bytes_feature(target_image),
        'target_image_temperature': _float_feature(target_image_temperature),
        'target_image_snow_depth': _float_feature(target_image_snow_depth),
        'target_image_precipitations': _serialized_bytes_feature(target_image_precipitations),
        'target_image_start_time': _bytes_feature(target_image_start_time.encode('utf-8')),
        'target_image_data_take_id': _bytes_feature(target_image_data_take_id.encode('utf-8')),
        'target_image_incidence_angle': _float_feature(target_image_incidence_angle),
        'target_image_platform_heading': _float_feature(target_image_platform_heading),
        'target_image_mission_id': _bytes_feature(target_image_mission_id.encode('utf-8')),

        'input_image_stack': _serialized_bytes_feature(input_image_stack),
        'input_image_temperatures': _serialized_bytes_feature(input_image_temperatures),
        'input_image_precipitations': _serialized_bytes_feature(input_image_precipitations),
        'input_image_snow_depths': _serialized_bytes_feature(input_image_snow_depths),
        'input_image_start_times': _serialized_bytes_feature(input_image_start_times),
        'input_image_data_take_ids': _serialized_bytes_feature(input_image_data_take_ids),
        'input_image_incidence_angles': _serialized_bytes_feature(input_image_incidence_angles),
        'input_image_platform_headings': _serialized_bytes_feature(input_image_platform_headings),
        'input_image_mission_ids': _serialized_bytes_feature(input_image_mission_ids),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Last entry in all array parameters is the target
def _serialize_example(
    image_location_wkt,
    dem_rast,
    forestmask,
    images,
    image_data_take_ids,
    image_start_times,
    incidence_angles,
    platform_headings,
    mission_ids,
    temperatures,
    precipitations,
    snow_depths,
):
    target_image = images[-2:]
    target_image_start_time = image_start_times[-1]
    target_image_data_take_id = image_data_take_ids[-1]
    target_image_incidence_angle = incidence_angles[-1]
    target_image_platform_heading = platform_headings[-1]
    target_image_temperature = temperatures[-1]
    target_image_precipitations = precipitations[-1]
    target_image_snow_depth = snow_depths[-1]
    target_image_mission_id = mission_ids[-1]

    input_image_stack = images[0:-2]
    input_image_start_times = image_start_times[0:-1]
    input_image_data_take_ids = image_data_take_ids[0:-1]
    input_image_incidence_angles = incidence_angles[0:-1]
    input_image_platform_headings = platform_headings[0:-1]
    input_image_temperatures = temperatures[0:-1]
    input_image_precipitations = precipitations[0:-1]
    input_image_snow_depths = snow_depths[0:-1]
    input_image_mission_ids = mission_ids[0:-1]

    return serialize_example(
        image_location_wkt,
        dem_rast,
        forestmask,
        target_image,
        target_image_start_time,
        target_image_data_take_id,
        target_image_incidence_angle,
        target_image_platform_heading,
        target_image_temperature,
        target_image_precipitations,
        target_image_snow_depth,
        target_image_mission_id,
        input_image_stack,
        input_image_start_times,
        input_image_data_take_ids,
        input_image_incidence_angles,
        input_image_platform_headings,
        input_image_temperatures,
        input_image_precipitations,
        input_image_snow_depths,
        input_image_mission_ids,
    )


feature_description = {
    'image_location': tf.io.FixedLenFeature([], tf.string),
    'dem_rast': tf.io.FixedLenFeature([], tf.string),
    'forestmask': tf.io.FixedLenFeature([], tf.string, default_value=tf.io.serialize_tensor(np.empty(0, dtype=np.single))),

    'target_image': tf.io.FixedLenFeature([], tf.string),
    'target_image_temperature': tf.io.FixedLenFeature([], tf.float32),
    'target_image_snow_depth': tf.io.FixedLenFeature([], tf.float32),
    'target_image_precipitations': tf.io.FixedLenFeature([], tf.string),
    'target_image_start_time': tf.io.FixedLenFeature([], tf.string),
    'target_image_data_take_id': tf.io.FixedLenFeature([], tf.string),
    'target_image_incidence_angle': tf.io.FixedLenFeature([], tf.float32),
    'target_image_platform_heading': tf.io.FixedLenFeature([], tf.float32),
    'target_image_mission_id': tf.io.FixedLenFeature([], tf.string, default_value=''),

    'input_image_stack': tf.io.FixedLenFeature([], tf.string),
    'input_image_temperatures': tf.io.FixedLenFeature([], tf.string),
    'input_image_precipitations': tf.io.FixedLenFeature([], tf.string),
    'input_image_snow_depths': tf.io.FixedLenFeature([], tf.string),
    'input_image_start_times': tf.io.FixedLenFeature([], tf.string),
    'input_image_data_take_ids': tf.io.FixedLenFeature([], tf.string),
    'input_image_incidence_angles': tf.io.FixedLenFeature([], tf.string),
    'input_image_platform_headings': tf.io.FixedLenFeature([], tf.string),
    'input_image_mission_ids': tf.io.FixedLenFeature([], tf.string, default_value=tf.io.serialize_tensor(np.empty(0, dtype=np.dtype(('U', 3))))),
}


def parse_dataset_with_simulated_change(filename, compression_type):
    raw_image_dataset = tf.data.TFRecordDataset(filename, compression_type=compression_type)

    def _parse_example_fn(example_proto):
        _feature_description = {
            **feature_description,
            'simulated_change_mask': tf.io.FixedLenFeature([], tf.string),
            'num_simulated_changes': tf.io.FixedLenFeature([], tf.int64),
            'img_superpixel_map': tf.io.FixedLenFeature([], tf.string),
            'img_superpixel_sets': tf.io.FixedLenFeature([], tf.string),
            'simulated_change_image': tf.io.FixedLenFeature([], tf.string),
            'target_image_prediction': tf.io.FixedLenFeature([], tf.string, default_value=tf.io.serialize_tensor(np.empty(0, dtype=np.single))),
        }
        example = tf.io.parse_single_example(example_proto, _feature_description)
        return {
            'image_location': example['image_location'],
            'dem_rast': tf.io.parse_tensor(example['dem_rast'], tf.float32),
            'forestmask': tf.io.parse_tensor(example['forestmask'], tf.float32),
            'target_image': tf.io.parse_tensor(example['target_image'], tf.float32),
            'target_image_temperature': example['target_image_temperature'],
            'target_image_snow_depth': example['target_image_snow_depth'],
            'target_image_precipitations': tf.io.parse_tensor(example['target_image_precipitations'], tf.float64),
            'target_image_start_time': example['target_image_start_time'],
            'target_image_data_take_id': example['target_image_data_take_id'],
            'target_image_incidence_angle': example['target_image_incidence_angle'],
            'target_image_platform_heading': example['target_image_platform_heading'],
            'target_image_mission_id': example['target_image_mission_id'],
            'input_image_stack': tf.io.parse_tensor(example['input_image_stack'], tf.float32),
            'input_image_temperatures': tf.io.parse_tensor(example['input_image_temperatures'], tf.float64),
            'input_image_precipitations': tf.io.parse_tensor(example['input_image_precipitations'], tf.float64),
            'input_image_snow_depths': tf.io.parse_tensor(example['input_image_snow_depths'], tf.float64),
            'input_image_start_times': tf.io.parse_tensor(example['input_image_start_times'], tf.string),
            'input_image_data_take_ids': tf.io.parse_tensor(example['input_image_data_take_ids'], tf.string),
            'input_image_incidence_angles': tf.io.parse_tensor(example['input_image_incidence_angles'], tf.float64),
            'input_image_platform_headings': tf.io.parse_tensor(example['input_image_platform_headings'], tf.float64),
            'input_image_mission_ids': tf.io.parse_tensor(example['input_image_mission_ids'], tf.string),
            'simulated_change_mask': tf.io.parse_tensor(example['simulated_change_mask'], tf.int32),
            'num_simulated_changes': example['num_simulated_changes'],
            'img_superpixel_map': tf.io.parse_tensor(example['img_superpixel_map'], tf.int64),
            'img_superpixel_sets': example['img_superpixel_sets'],
            'target_image_prediction': tf.io.parse_tensor(example['target_image_prediction'], tf.float32),
            'simulated_change_image': tf.io.parse_tensor(example['simulated_change_image'], tf.float32),
        }

    raw_image_dataset = raw_image_dataset.map(_parse_example_fn)

    return raw_image_dataset


def parse_dataset(filename, compression_type):
    raw_image_dataset = tf.data.TFRecordDataset(filename, compression_type=compression_type)

    def _parse_example_fn(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        return {
            'image_location': example['image_location'],
            'dem_rast': tf.io.parse_tensor(example['dem_rast'], tf.float32),
            'forestmask': tf.io.parse_tensor(example['forestmask'], tf.float32),
            'target_image': tf.io.parse_tensor(example['target_image'], tf.float32),
            'target_image_temperature': example['target_image_temperature'],
            'target_image_snow_depth': example['target_image_snow_depth'],
            'target_image_precipitations': tf.io.parse_tensor(example['target_image_precipitations'], tf.float64),
            'target_image_start_time': example['target_image_start_time'],
            'target_image_data_take_id': example['target_image_data_take_id'],
            'target_image_incidence_angle': example['target_image_incidence_angle'],
            'target_image_platform_heading': example['target_image_platform_heading'],
            'target_image_mission_id': example['target_image_mission_id'],
            'input_image_stack': tf.io.parse_tensor(example['input_image_stack'], tf.float32),
            'input_image_temperatures': tf.io.parse_tensor(example['input_image_temperatures'], tf.float64),
            'input_image_precipitations': tf.io.parse_tensor(example['input_image_precipitations'], tf.float64),
            'input_image_snow_depths': tf.io.parse_tensor(example['input_image_snow_depths'], tf.float64),
            'input_image_start_times': tf.io.parse_tensor(example['input_image_start_times'], tf.string),
            'input_image_data_take_ids': tf.io.parse_tensor(example['input_image_data_take_ids'], tf.string),
            'input_image_incidence_angles': tf.io.parse_tensor(example['input_image_incidence_angles'], tf.float64),
            'input_image_platform_headings': tf.io.parse_tensor(example['input_image_platform_headings'], tf.float64),
            'input_image_mission_ids': tf.io.parse_tensor(example['input_image_mission_ids'], tf.string),
        }

    raw_image_dataset = raw_image_dataset.map(_parse_example_fn)

    return raw_image_dataset

def normalize(ds_stats):
    def min_max(min, max, feature, a=0.0, b=1.0, clip=False):
        result = a + ((feature - min) * (b - a) / (max - min))
        if clip:
            if tf.is_tensor(result):
                result = tf.clip_by_value(result, min, max)
            else:
                result = np.clip(result, min, max)
        return result
    def std_norm(mean, var, feature):
        std_dev = np.sqrt(var)
        return (feature - mean) / std_dev
    def f(sample):
        img_std_dev = np.sqrt(ds_stats['sar_images']['var'])
        img_mean = ds_stats['sar_images']['mean']
        img_min = img_mean - (img_std_dev * 4)
        img_max = img_mean + (img_std_dev * 4)
        return {
            'dem_rast': min_max(0, ds_stats['dem_rast']['max'], sample['dem_rast']),
            'target_image': min_max(img_min, img_max, sample['target_image'], a=-1.0, b=1.0, clip=True),
            'forestmask': sample['forestmask'], # note that this is 1, 0 mask, so no normalization
            'input_image_stack': min_max(img_min, img_max, sample['input_image_stack'], a=-1.0, b=1.0, clip=True),
            'input_image_snow_depths': min_max(0, ds_stats['snow_depths']['max'], sample['input_image_snow_depths']),
            'target_image_snow_depth': min_max(0, ds_stats['snow_depths']['max'], sample['target_image_snow_depth']),
            'input_image_precipitations': min_max(0, ds_stats['precipitations']['max'], sample['input_image_precipitations']),
            'target_image_precipitations': min_max(0, ds_stats['precipitations']['max'], sample['target_image_precipitations']),
            'input_image_temperatures': min_max(-30.0, 30.0, sample['input_image_temperatures'], a=-1.0, b=1.0),
            'target_image_temperature': min_max(-30.0, 30.0, sample['target_image_temperature'], a=-1.0, b=1.0),
            'input_image_platform_headings': sample['input_image_platform_headings'] / 360.0,
            'target_image_platform_heading': sample['target_image_platform_heading'] / 360.0,
            # The angle should be between 29.1 and 46 degrees accoring to this
            # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/acquisition-modes/interferometric-wide-swath
            'input_image_incidence_angles': min_max(29.0, 46.0, sample['input_image_incidence_angles']),
            'target_image_incidence_angle': min_max(29.0, 46.0, sample['target_image_incidence_angle']),
            'input_image_mission_ids': sample['input_image_mission_ids'],
            'target_image_mission_id': sample['target_image_mission_id'],
        }
    return f


def x_y_split(sample):
    target_mission_id = tf.cast(sample['target_image_mission_id'] == 'S1A', tf.float64)
    input_mission_ids = tf.map_fn(
        lambda x: tf.cast(x == 'S1A', tf.float64),
        sample['input_image_mission_ids'],
        fn_output_signature=tf.float64,
    )
    # NOTE! if you change this, also update ignore_weather_info function
    latent_metadata = tf.concat([
        tf.reshape(sample['input_image_temperatures'], [-1]),
        tf.reshape(sample['input_image_snow_depths'], [-1]),
        tf.reshape(sample['input_image_platform_headings'], [-1]),
        tf.reshape(sample['input_image_incidence_angles'], [-1]),
        tf.reshape(sample['input_image_precipitations'], [-1]),
        tf.reshape(input_mission_ids, [-1]),
        [
            sample['target_image_temperature'],
            sample['target_image_snow_depth'],
            sample['target_image_platform_heading'],
            sample['target_image_incidence_angle'],
            target_mission_id,
        ],
        tf.reshape(sample['target_image_precipitations'], [-1]),
    ], axis=-1)
    input_image_stack = tf.concat([
        sample['input_image_stack'],
        tf.expand_dims(sample['dem_rast'], axis=0),
    ], axis=0)

    # Channel last format
    input_image_stack = tf.transpose(input_image_stack, perm=[1, 2, 0])
    target_image = tf.transpose(sample['target_image'], perm=[1, 2, 0])

    d = {
        'input_image_stack': input_image_stack,
        'latent_metadata': latent_metadata,
        'target_image': target_image,
        'forestmask': sample['forestmask'],
        'target_image': target_image,
    }

    return d

# Hacky way of removing the weather data from the latent vector to experiment
# how model works without them.
def ignore_weather_info(latent_vector_batch):
    num_imgs_per_date = 4
    num_percipitations_per_date = 4
    # skip temperatures and snow_depths
    input_platform_headings_slice = slice(num_imgs_per_date * 2, num_imgs_per_date * 3)
    input_incidence_angles_slice = slice(num_imgs_per_date * 3, num_imgs_per_date * 4)
    percipitation_offset = num_imgs_per_date * 4 + num_imgs_per_date * num_percipitations_per_date
    input_mission_ids_slice = slice(percipitation_offset, percipitation_offset + num_imgs_per_date)

    target_data_offset = percipitation_offset + num_imgs_per_date

    target_platform_heading_idx = target_data_offset + 2
    target_incidence_angle_idx = target_data_offset + 3
    target_mission_id_idx = target_data_offset + 4

    input_platform_headings = latent_vector_batch[:, input_platform_headings_slice]
    input_incidence_angles = latent_vector_batch[:, input_incidence_angles_slice]
    input_mission_ids = latent_vector_batch[:, input_mission_ids_slice]

    target_platform_heading = latent_vector_batch[:, target_platform_heading_idx]
    target_incidence_angle = latent_vector_batch[:, target_incidence_angle_idx]
    target_mission_id = latent_vector_batch[:, target_mission_id_idx]

    return (
        input_platform_headings,
        input_incidence_angles,
        input_mission_ids,
        target_platform_heading,
        target_incidence_angle,
        target_mission_id,
    )
