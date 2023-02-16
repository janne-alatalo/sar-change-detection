import multiprocessing
import datetime
import argparse
import pathlib
import glob
import json
import sys
import re
import os

# Disable cuda because we don't need it
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import psycopg2
import psycopg2.extras
import numpy as np
from osgeo import gdal, ogr, osr
import tensorflow as tf

from simulated_change import add_statistical_change
from dataset_functions import serialize_simulated_change_example, parse_dataset


def to_image(img_arr):
    img_max = np.nanmax(img_arr)
    img_min = np.nanmin(img_arr)

    img_arr = (img_arr - img_min) / (img_max - img_min)
    third_chan = img_arr[:, :, 1] / img_arr[:, :, 0]

    third_chan_max = np.nanmax(third_chan)
    third_chan_min = np.nanmin(third_chan)
    third_chan = (third_chan - third_chan_min) / (third_chan_max - third_chan_min)

    full_img = np.stack(
        [
            img_arr[:, :, 0],
            img_arr[:, :, 1],
            third_chan,
        ], axis=-1
    )

    return full_img


def process_sample(params):
    conn = psycopg2.connect(
            dbname=args.dbname,
            user=args.dbuser,
            password=args.dbpassword,
            host=args.dbhost,
            port=args.dbport,
            cursor_factory=psycopg2.extras.DictCursor,
        )
    conn.set_session(readonly=True, autocommit=True)
    cursor = conn.cursor()
    cursor.execute("select true")
    [
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
    ] = params

    area_size_meters = args.area_size_meters * 1.1

    wgs80_srs = osr.SpatialReference()
    wgs80_srs.ImportFromEPSG(4326)
    fin_srs = osr.SpatialReference()
    fin_srs.ImportFromEPSG(3067)

    point_str = image_location
    point = ogr.CreateGeometryFromWkt(point_str)
    point.AssignSpatialReference(wgs80_srs)
    location_epsg4326 = point.Clone()
    location_epsg4326.FlattenTo2D()
    point.TransformTo(fin_srs)

    position = (location_epsg4326.GetY(), location_epsg4326.GetX())

    area_size_meters = args.area_size_meters
    offset = area_size_meters / 2

    target_image_transposed = np.transpose(target_image, axes=[1, 2, 0])
    x_size = target_image_transposed.shape[1]
    y_size = target_image_transposed.shape[0]
    geotransform = [
        point.GetX() - offset,
        area_size_meters / x_size,
        0,
        point.GetY() + offset,
        0,
        -area_size_meters / y_size,
    ]

    (
        simulated_change_image,
        num_changes,
        change_mask_arr,
        img_superpixel_map,
        img_superpixel_sets,
        _,
    ) = add_statistical_change(
        cursor,
        image_location,
        target_image_data_take_id,
        target_image_start_time,
        target_image,
        (0, 3),
        args.area_size_meters,
    )

    img_superpixel_sets_json = json.dumps(img_superpixel_sets)
    serialized_sample = serialize_simulated_change_example(
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
        num_changes,
        change_mask_arr,
        img_superpixel_map,
        img_superpixel_sets_json,
        simulated_change_image,
    )

    return serialized_sample


def main(args):
    tfrecord_files = [f for tfrecord_glob in args.tfrecord_files for f in glob.glob(tfrecord_glob)]
    tfrecord_files.sort()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if len(tfrecord_files) == 0:
        print(f'No files found in glob(s) {args.tfrecord_files}', file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(
            dbname=args.dbname,
            user=args.dbuser,
            password=args.dbpassword,
            host=args.dbhost,
            port=args.dbport,
            cursor_factory=psycopg2.extras.DictCursor,
        )
    conn.set_session(readonly=True, autocommit=True)
    cursor = conn.cursor()
    cursor.execute("select true")

    tfrecord_options = tf.io.TFRecordOptions(
        compression_type=args.tfrecord_compression,
    )
    pool_size = args.num_parallel if args.num_parallel is not None else multiprocessing.cpu_count()
    process_pool = None
    if pool_size > 1:
        process_pool = multiprocessing.Pool(pool_size)

    for f in tfrecord_files:
        f_path = pathlib.Path(f)
        target_tfrecord_path = output_dir / f_path.name
        ds = parse_dataset(f, args.tfrecord_compression)
        pickleable_ds = map(
            lambda s: [
                s['image_location'].numpy().decode(),
                s['dem_rast'].numpy(),
                s['forestmask'].numpy(),
                s['target_image'].numpy(),
                s['target_image_start_time'].numpy().decode(),
                s['target_image_data_take_id'].numpy().decode(),
                s['target_image_incidence_angle'].numpy(),
                s['target_image_platform_heading'].numpy(),
                s['target_image_temperature'].numpy(),
                s['target_image_precipitations'].numpy(),
                s['target_image_snow_depth'].numpy(),
                s['target_image_mission_id'].numpy().decode(),
                s['input_image_stack'].numpy(),
                s['input_image_start_times'].numpy(),
                s['input_image_data_take_ids'].numpy(),
                s['input_image_incidence_angles'].numpy(),
                s['input_image_platform_headings'].numpy(),
                s['input_image_temperatures'].numpy(),
                s['input_image_precipitations'].numpy(),
                s['input_image_snow_depths'].numpy(),
                s['input_image_mission_ids'].numpy(),
            ], ds
        )

        with tf.io.TFRecordWriter(str(target_tfrecord_path), options=tfrecord_options) as writer:
            if process_pool is not None:
                for i, serialized_sample in enumerate(process_pool.imap(process_sample, pickleable_ds)):
                    writer.write(serialized_sample)
                    print(f'{i} sample')
            else:
                for i, sample in enumerate(pickleable_ds):
                    serialized_sample = process_sample(sample)
                    writer.write(serialized_sample)
                    print(f'{i} sample')



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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dbname",
        type=str,
        default="forestdamage",
        help="Database name",
    )
    parser.add_argument(
        "--dbuser",
        type=str,
        default="postgres",
        help="Database user",
    )
    parser.add_argument(
        "--dbpassword",
        type=str,
        required=True,
        help="Database password",
    )
    parser.add_argument(
        "--dbhost",
        type=str,
        required=True,
        help="Database host",
    )
    parser.add_argument(
        "--dbport",
        type=int,
        default=5432,
        help="Database port",
    )
    parser.add_argument(
        "--area_size_meters",
        type=int,
        default=3000,
        help="How big square the sample area is in meters",
    )
    parser.add_argument(
        "--num_parallel",
        type=int,
        default=None,
        help="Parallel processing pool size",
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='simulated_change',
        help='Dataset statistics that are used for normalization',
    )
    parser.add_argument(
        "--tfrecord_compression",
        default="GZIP",
        choices=["GZIP", "ZLIB", ""],
        help="TFRecord compression type (set to \"\" to use no compression)",
    )
    parser.add_argument(
        'tfrecord_files',
        type=str,
        nargs='+',
        help='TFRecord files glob',
    )

    args = parser.parse_args()

    main(args)
