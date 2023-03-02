import argparse
import pathlib
import os
import logging
import json
import os

# Disable cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from dataset_functions import parse_dataset, _serialized_bytes_feature, x_y_split, normalize


def main(args):

    with open(args.dataset_stats, 'r') as f:
        ds_stats = json.load(f)

    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_path = pathlib.Path(args.filename)
    _path = input_path
    path_stem = _path.stem
    while _path.suffixes:
        path_stem = _path.stem
        _path = pathlib.Path(path_stem)

    output_filename = f"{path_stem}.tfrecord"
    if args.dst_tfrecord_compression != "":
        output_filename += f".{args.dst_tfrecord_compression}"

    output_filepath = output_path / output_filename

    ds = parse_dataset(args.filename, args.src_tfrecord_compression)
    ds = ds.map(normalize(ds_stats))
    ds = ds.map(x_y_split)
    tfrecord_options = tf.io.TFRecordOptions(
        compression_type=args.dst_tfrecord_compression,
    )
    with tf.io.TFRecordWriter(os.fspath(output_filepath), options=tfrecord_options) as writer:

        for sample in ds:
            feature = {
                'input_image_stack': _serialized_bytes_feature(sample['input_image_stack']),
                'latent_metadata': _serialized_bytes_feature(sample['latent_metadata']),
                'target_image': _serialized_bytes_feature(sample['target_image']),
                'forestmask': _serialized_bytes_feature(sample['forestmask']),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            example = example_proto.SerializeToString()
            writer.write(example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_tfrecord_compression",
        default="GZIP",
        choices=["GZIP", "ZLIB", ""],
        help="TFRecord compression type (set to \"\" to use no compression)",
    )
    parser.add_argument(
        "--dst_tfrecord_compression",
        default="GZIP",
        choices=["GZIP", "ZLIB", ""],
        help="TFRecord compression type (set to \"\" to use no compression)",
    )
    parser.add_argument(
        "--output_dir",
        default="preprocessed",
        help="Output directory for the new tfrecords",
    )
    parser.add_argument(
        '--dataset_stats',
        type=str,
        default='stats.json',
        help='Dataset statistics that are used for normalization',
    )
    parser.add_argument(
        "filename",
        type=str,
        help="input filename",
    )

    args = parser.parse_args()

    main(args)
