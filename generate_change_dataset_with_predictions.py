import datetime
import pathlib
import glob
import json
import sys
import re
import os

import configargparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from dataset_functions import parse_dataset_with_simulated_change, x_y_split, normalize, serialize_simulated_change_example
from dataset import denormalize_imgs


def main(args):

    with open(args.dataset_stats, 'r') as f:
        ds_stats = json.load(f)

    tfrecord_files = [f for tfrecord_glob in args.tfrecord_files for f in glob.glob(tfrecord_glob)]
    tfrecord_files.sort()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if len(tfrecord_files) == 0:
        print(f'No files found in glob(s) {args.tfrecord_files}', file=sys.stderr)
        sys.exit(1)

    orig_ds = parse_dataset_with_simulated_change(tfrecord_files, args.tfrecord_compression)
    processed_ds = orig_ds.map(normalize(ds_stats), deterministic=True)
    processed_ds = processed_ds.map(x_y_split, deterministic=True)
    processed_ds = processed_ds.batch(args.batch_size, deterministic=True)
    orig_ds = orig_ds.batch(args.batch_size, deterministic=True)

    model = tf.keras.models.load_model(args.model_checkpoint, compile=True)

    filename_digits = 4
    i = 0

    tfrecord_options = tf.io.TFRecordOptions(
        compression_type=args.tfrecord_compression,
    )
    filename_n = 0
    filename = f"{str(filename_n).zfill(filename_digits)}.tfrecord"
    if args.tfrecord_compression != "":
        filename += f".{args.tfrecord_compression}"
    filepath = output_dir / filename
    writer = tf.io.TFRecordWriter(os.fspath(filepath), options=tfrecord_options)
    for processed_batch, orig_batch in zip(processed_ds, orig_ds):
        batch_prediction = model(processed_batch)
        batch_prediction = denormalize_imgs(batch_prediction, ds_stats)
        sample_features = [
            orig_batch['image_location'],
            orig_batch['dem_rast'],
            orig_batch['forestmask'],
            orig_batch['target_image'],
            orig_batch['target_image_start_time'],
            orig_batch['target_image_data_take_id'],
            orig_batch['target_image_incidence_angle'],
            orig_batch['target_image_platform_heading'],
            orig_batch['target_image_temperature'],
            orig_batch['target_image_precipitations'],
            orig_batch['target_image_snow_depth'],
            orig_batch['target_image_mission_id'],
            orig_batch['input_image_stack'],
            orig_batch['input_image_start_times'],
            orig_batch['input_image_data_take_ids'],
            orig_batch['input_image_incidence_angles'],
            orig_batch['input_image_platform_headings'],
            orig_batch['input_image_temperatures'],
            orig_batch['input_image_precipitations'],
            orig_batch['input_image_snow_depths'],
            orig_batch['input_image_mission_ids'],
            orig_batch['num_simulated_changes'],
            orig_batch['simulated_change_mask'],
            orig_batch['img_superpixel_map'],
            orig_batch['img_superpixel_sets'],
            orig_batch['simulated_change_image'],
        ]
        for z in zip(batch_prediction, *sample_features):
            if i >= args.num_samples_per_file:
                writer.close()
                filename_n += 1
                filename = f"{str(filename_n).zfill(filename_digits)}.tfrecord"
                if args.tfrecord_compression != "":
                    filename += f".{args.tfrecord_compression}"
                filepath = output_dir / filename
                writer = tf.io.TFRecordWriter(os.fspath(filepath), options=tfrecord_options)
                i = 0
            i += 1
            (
                y,
                image_location,
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
                img_superpixel_sets,
                simulated_change_image,
            ) = z
            serialized = serialize_simulated_change_example(
                image_location.numpy().decode(),
                dem_rast.numpy(),
                forestmask.numpy(),
                target_image.numpy(),
                target_image_start_time.numpy().decode(),
                target_image_data_take_id.numpy().decode(),
                target_image_incidence_angle.numpy(),
                target_image_platform_heading.numpy(),
                target_image_temperature.numpy(),
                target_image_precipitations.numpy(),
                target_image_snow_depth.numpy(),
                target_image_mission_id.numpy().decode(),
                input_image_stack.numpy(),
                input_image_start_times.numpy(),
                input_image_data_take_ids.numpy(),
                input_image_incidence_angles.numpy(),
                input_image_platform_headings.numpy(),
                input_image_temperatures.numpy(),
                input_image_precipitations.numpy(),
                input_image_snow_depths.numpy(),
                input_image_mission_ids.numpy(),
                num_simulated_changes.numpy(),
                simulated_change_mask.numpy(),
                img_superpixel_map.numpy(),
                img_superpixel_sets.numpy().decode(),
                simulated_change_image.numpy(),
                target_image_prediction=y.numpy(),
            )
            writer.write(serialized)


if __name__ == '__main__':

    def coordinate(arg_value, pat=re.compile(r'\d+\.\d+,\d+\.\d+')):
        if not pat.match(arg_value):
            raise argparse.ArgumentTypeError
        longitude, latitude = arg_value.split(',')
        return float(longitude), float(latitude)

    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def date_arg(value):
        try:
            return datetime.datetime.strptime(value, '%Y-%m-%d')
        except ValueError:
            raise argparse.ArgumentTypeError(f'Invalid date argument {value}')

    parser = configargparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        default='simulated_change_with_prediction',
        help='Dataset statistics that are used for normalization',
    )
    parser.add_argument(
        "--tfrecord_compression",
        default="GZIP",
        choices=["GZIP", "ZLIB", ""],
        help="TFRecord compression type (set to \"\" to use no compression)",
    )
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        required=True,
        help='Model checkpoint directory.',
    )
    parser.add_argument(
        "--num_samples_per_file",
        type=int,
        default=100,
        help="How many samples per file",
    )
    parser.add_argument(
        '--batch_size',
        env_var='BATCH_SIZE',
        type=int,
        default=100,
        help='Batch size',
    )
    parser.add_argument(
        '--dataset_stats',
        env_var='DATASET_STATS',
        type=str,
        default='stats.json',
        help='Dataset statistics that are used for normalization',
    )
    parser.add_argument(
        'tfrecord_files',
        type=str,
        nargs='+',
        help='TFRecord files glob',
    )

    args = parser.parse_args()

    main(args)
