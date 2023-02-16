import tensorflow as tf
import numpy as np

feature_description = {
    'input_image_stack': tf.io.FixedLenFeature([], tf.string),
    'latent_metadata': tf.io.FixedLenFeature([], tf.string),
    'target_image': tf.io.FixedLenFeature([], tf.string),
    'forestmask': tf.io.FixedLenFeature([], tf.string),
}


def _parse_example_fn(example_proto):
    example = tf.io.parse_example(example_proto, feature_description)
    def _parse_tensor_float64(t):
        return tf.io.parse_tensor(t, tf.float64)
    def _parse_tensor_float32(t):
        return tf.io.parse_tensor(t, tf.float32)
    target_image = tf.vectorized_map(_parse_tensor_float32, example['target_image'])
    parsed = {
        'input_image_stack': tf.vectorized_map(_parse_tensor_float32, example['input_image_stack']),
        'latent_metadata': tf.vectorized_map(_parse_tensor_float64, example['latent_metadata']),
        'target_image': target_image,
        'forestmask': tf.vectorized_map(_parse_tensor_float32, example['forestmask']),
    }
    return parsed, target_image


def get_dataset(files, compression_type='GZIP', num_parallel_reads=1, batch_size=100):
    ds = tf.data.TFRecordDataset(
        files,
        compression_type=compression_type,
        num_parallel_reads=num_parallel_reads,
        buffer_size=100_000_000,
    )
    ds = ds.batch(
        batch_size,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.map(
        _parse_example_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return ds


def denormalize_imgs(imgs, ds_stats):
    img_std_dev = np.sqrt(ds_stats['sar_images']['var'])
    img_mean = ds_stats['sar_images']['mean']
    img_min = img_mean - (img_std_dev * 4)
    img_max = img_mean + (img_std_dev * 4)
    a = -1.0
    b = 1.0

    result = (((imgs - a) * (img_max - img_min)) / (b - a)) + img_min

    return result
