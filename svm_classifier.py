import multiprocessing
import datetime
import argparse
import pathlib
import random
import glob
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import configargparse
import tensorflow as tf
import numpy as np
import git

from dataset_functions import parse_dataset_with_simulated_change


feature_description = {
    'target_image': tf.io.FixedLenFeature([], tf.string),
    'target_image_prediction': tf.io.FixedLenFeature([], tf.string),
    'closest_input': tf.io.FixedLenFeature([], tf.string),
    'simulated_change_mask': tf.io.FixedLenFeature([], tf.string),
    'num_simulated_changes': tf.io.FixedLenFeature([], tf.int64),
    'img_superpixel_map': tf.io.FixedLenFeature([], tf.string),
    'img_superpixel_sets': tf.io.FixedLenFeature([], tf.string),
}


def parse_example(example_proto):
    sample = tf.io.parse_example(example_proto, feature_description)
    parsed = {
        'target_image': tf.io.parse_tensor(sample['target_image'], tf.float32),
        'target_image_prediction': tf.io.parse_tensor(sample['target_image_prediction'], tf.float32),
        'closest_input': tf.io.parse_tensor(sample['closest_input'], tf.float32),
        'simulated_change_mask': tf.io.parse_tensor(sample['simulated_change_mask'], tf.int32),
        'num_simulated_changes': sample['num_simulated_changes'],
        'img_superpixel_map': tf.io.parse_tensor(sample['img_superpixel_map'], tf.int64),
        'img_superpixel_sets_json': sample['img_superpixel_sets'],
    }
    return parsed


def simulated_change_diff_samples(params):
    (
        orig_target_image,
        target_image_incidence_angle,
        target_image_platform_heading,
        input_image_stack,
        input_image_incidence_angles,
        input_image_platform_headings,
        simulated_change_mask,
        num_simulated_changes,
        img_superpixel_map,
        img_superpixel_sets_json,
        target_image_prediction,
        simulated_change_image,
    ) = params

    # ia == incidence angle, ph = platform heading, t = target, i = input
    ia_max = np.max(np.append(target_image_incidence_angle, input_image_incidence_angles))
    ia_min = np.min(np.append(target_image_incidence_angle, input_image_incidence_angles))
    ph_max = np.max(np.append(target_image_platform_heading, input_image_platform_headings))
    ph_min = np.min(np.append(target_image_platform_heading, input_image_platform_headings))

    t_ia = (target_image_incidence_angle - ia_min) / (ia_max - ia_min)
    t_ph = (target_image_platform_heading - ph_min) / (ph_max - ph_min)
    i_ia = (input_image_incidence_angles - ia_min) / (ia_max - ia_min)
    i_ph = (input_image_platform_headings - ph_min) / (ph_max - ph_min)

    t_v = np.stack([t_ia, t_ph])
    i_v = np.transpose(np.stack([i_ia, i_ph]))

    v_dist = np.sqrt(np.sum(np.square(t_v - i_v), axis=1))
    # 2 channels per image -> first channel index is n * 2
    n = np.argmin(v_dist) * 2
    closest_input = input_image_stack[n:n + 2]
    closest_input = np.transpose(closest_input, [1, 2, 0])

    if args.simulated_change_shift is not None:
        target_image = np.transpose(orig_target_image, [1, 2, 0])
        shift_mask = np.zeros_like(target_image)
        change_mask_bool = simulated_change_mask != 0.0
        shift_mask[change_mask_bool, :] = args.simulated_change_shift
        simulated_change_image = target_image + shift_mask


    if args.di_method == '-':
        prediction_diff = simulated_change_image - target_image_prediction
        previous_diff = simulated_change_image - closest_input
    elif args.di_method == '/':
        max_val = np.max([closest_input, target_image_prediction, simulated_change_image])
        min_val = np.min([closest_input, target_image_prediction, simulated_change_image])
        target_image_prediction_norm = (target_image_prediction - min_val) / (max_val - min_val) + 0.01
        simulated_change_image_norm = (simulated_change_image - min_val) / (max_val - min_val) + 0.01
        closest_input_norm = (closest_input - min_val) / (max_val - min_val) + 0.01
        prediction_diff = simulated_change_image_norm / target_image_prediction_norm
        previous_diff = simulated_change_image_norm / closest_input_norm
    elif args.di_method == 'log':
        max_val = np.max([closest_input, target_image_prediction, simulated_change_image])
        min_val = np.min([closest_input, target_image_prediction, simulated_change_image])
        target_image_prediction_norm = (target_image_prediction - min_val) / (max_val - min_val) + 0.01
        simulated_change_image_norm = (simulated_change_image - min_val) / (max_val - min_val) + 0.01
        closest_input_norm = (closest_input - min_val) / (max_val - min_val) + 0.01
        prediction_diff = np.log(simulated_change_image_norm / target_image_prediction_norm)
        previous_diff = np.log(simulated_change_image_norm / closest_input_norm)

    y_mask = (simulated_change_mask != 0).astype(dtype=np.float32)

    return (prediction_diff, previous_diff, y_mask)


def sample_generator(ds):
    def g():
        for img_sample in ds:
            prediction_diff, previous_diff, is_change_pixel = img_sample
            x_size, y_size = is_change_pixel.shape[0:2]
            change_indices = np.where(is_change_pixel == 1.0)
            for x, y in zip(*change_indices):
                datapoint = np.array(
                    [
                        *prediction_diff[x, y],
                        *previous_diff[x, y],
                        is_change_pixel[x, y]
                    ]
                )
                yield datapoint
                # for every change pixel yield also one non change pixel
                while True:
                    x_rand = random.randint(0, x_size - 1)
                    y_rand = random.randint(0, y_size - 1)
                    if is_change_pixel[x_rand, y_rand] != 1.0:
                        datapoint = np.array(
                            [
                                *prediction_diff[x_rand, y_rand],
                                *previous_diff[x_rand, y_rand],
                                is_change_pixel[x_rand, y_rand]
                            ]
                        )
                        yield datapoint
                        break
    return g


def main(args):
    tfrecord_files = [f for tfrecord_glob in args.tfrecord_files for f in glob.glob(tfrecord_glob)]
    tfrecord_files.sort()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    if repo.is_dirty() and not args.ignore_dirty_repo:
        print('Repository is dirty commit changes or pass --ignore_dirty_repo!')
        exit(1)

    if args.logs_tag is None:
        datenow_tag = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        tag_prefix = datenow_tag
    else:
        tag_prefix = args.logs_tag
    tag = tag_prefix
    logs_path = os.path.join('./logs', tag)
    logs_path_obj = pathlib.Path(logs_path)
    i = 1
    while True:
        try:
            logs_path_obj.mkdir(parents=True)
            break
        except FileExistsError:
            tag = f'{tag_prefix}_{i}'
            logs_path = os.path.join('./logs', tag)
            logs_path_obj = pathlib.Path(logs_path)
            i += 1

    train_args = json.dumps(
        {
            'tfrecord_files': tfrecord_files,
            'repo_is_dirty': repo.is_dirty(),
            'repo_commit': sha,
            'logs_path': os.path.abspath(logs_path),
            'di_method': args.di_method,
            'simulated_change_shift': args.simulated_change_shift,
        },
        indent=4,
    )
    print('==== TRAIN arguments ====')
    print(train_args)
    print('=========================')

    pool_size = args.num_parallel if args.num_parallel is not None else multiprocessing.cpu_count()
    process_pool = multiprocessing.Pool(pool_size)
    ds = parse_dataset_with_simulated_change(tfrecord_files, args.tfrecord_compression)
    ds = map(
        lambda x: (
                x['target_image'].numpy(),
                x['target_image_incidence_angle'].numpy(),
                x['target_image_platform_heading'].numpy(),
                x['input_image_stack'].numpy(),
                x['input_image_incidence_angles'].numpy(),
                x['input_image_platform_headings'].numpy(),
                x['simulated_change_mask'].numpy(),
                x['num_simulated_changes'].numpy(),
                x['img_superpixel_map'].numpy(),
                x['img_superpixel_sets'].numpy(),
                x['target_image_prediction'].numpy(),
                x['simulated_change_image'].numpy(),
            ),
        ds,
    )
    image_samples = list(process_pool.imap_unordered(simulated_change_diff_samples, ds))
    random.shuffle(image_samples)
    num_imgs = len(image_samples)
    test_dataset_size = int(np.round(num_imgs * 0.33))

    test_samples = image_samples[0:test_dataset_size]
    train_samples = image_samples[test_dataset_size:]

    print(f'Num train images {num_imgs - test_dataset_size}')
    print(f'Num test images {test_dataset_size}')

    train_arr = np.array(list(sample_generator(train_samples)()))
    test_arr = np.array(list(sample_generator(test_samples)()))
    x_train = train_arr[:, 0:4]
    y_train = train_arr[:, 4]

    x_test = test_arr[:, 0:4]
    y_test = test_arr[:, 4]

    x_train_pred_diff = x_train[:, 0:2]
    x_train_prev_diff = x_train[:, 2:4]
    x_test_pred_diff = x_test[:, 0:2]
    x_test_prev_diff = x_test[:, 2:4]

    pred_scaler = StandardScaler()
    x_train_pred_diff = pred_scaler.fit_transform(x_train_pred_diff)
    x_test_pred_diff = pred_scaler.transform(x_test_pred_diff)

    prev_scaler = StandardScaler()
    x_train_prev_diff = prev_scaler.fit_transform(x_train_prev_diff)
    x_test_prev_diff = prev_scaler.transform(x_test_prev_diff)

    svc_pred_diff = LinearSVC()
    svc_prev_diff = LinearSVC()

    svc_pred_diff.fit(x_train_pred_diff, y_train)
    svc_prev_diff.fit(x_train_prev_diff, y_train)

    pred_classification_report = classification_report(y_test, svc_pred_diff.predict(x_test_pred_diff))
    prev_classification_report = classification_report(y_test, svc_prev_diff.predict(x_test_prev_diff))

    print('========================== pred diff ==============================')
    print(pred_classification_report)
    print('========================== prev diff ==============================')
    print(prev_classification_report)
    print('===================================================================')

    pred_report_file = logs_path_obj / 'pred_classification_report.txt'
    prev_report_file = logs_path_obj / 'prev_classification_report.txt'
    with pred_report_file.open('w') as f:
        f.write(pred_classification_report)
    with prev_report_file.open('w') as f:
        f.write(prev_classification_report)


if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = configargparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        default='preprocessed_simulated_change',
        help='Output directory for the files',
    )
    parser.add_argument(
        "--tfrecord_compression",
        default="GZIP",
        choices=["GZIP", "ZLIB", ""],
        help="TFRecord compression type (set to \"\" to use no compression)",
    )
    parser.add_argument(
        "--num_samples_per_file",
        type=int,
        default=100,
        help="How many samples per file",
    )
    parser.add_argument(
        '--ignore_dirty_repo',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Ignore dirty git repository',
    )
    parser.add_argument(
        "--num_parallel",
        type=int,
        default=None,
        help="Parallel processing pool size",
    )
    parser.add_argument(
        '--di_method',
        type=str,
        default='-',
        choices=['-', '/', 'log'],
        help='Difference image computation method',
    )
    parser.add_argument(
        '--simulated_change_shift',
        type=float,
        default=None,
        help='Use shift simulated change method and add the argument value to the pixel values',
    )
    parser.add_argument(
        '--logs_tag',
        env_var='LOGS_TAG',
        type=str,
        default=None,
        help='Directory name for the logs',
    )
    parser.add_argument(
        'tfrecord_files',
        type=str,
        nargs='+',
        help='TFRecord files glob',
    )

    args = parser.parse_args()

    main(args)
