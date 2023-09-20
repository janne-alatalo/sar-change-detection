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
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
import configargparse
import tensorflow as tf
import numpy as np
import git
import matplotlib.pyplot as plt

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


def filter_samples(params):
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
        target_image_start_time,
        input_image_start_times,
    ) = params
    input_image_orbits = np.abs(input_image_platform_headings) > 90.0
    target_image_orbit = np.abs(target_image_platform_heading) > 90.0
    if target_image_orbit not in input_image_orbits:
        return False

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
    closest_input_idx = np.argmin(v_dist)

    angle_diff = target_image_incidence_angle - input_image_incidence_angles[closest_input_idx]
    if np.abs(angle_diff) > 1.0:
        return False

    return True


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
        target_image_start_time,
        input_image_start_times,
    ) = params

    if args.prev_selection_method == '1':
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
        closest_input_idx = np.argmin(v_dist)
        n = closest_input_idx * 2
        closest_input = input_image_stack[n:n + 2]
        closest_input = np.transpose(closest_input, [1, 2, 0])
    elif args.prev_selection_method == '2':
        input_image_orbits = np.abs(input_image_platform_headings) > 90.0
        target_image_orbit = np.abs(target_image_platform_heading) > 90.0

        closest_input_idx = np.max(np.argwhere(input_image_orbits == target_image_orbit))
        # 2 channels per image -> first channel index is n * 2
        n = closest_input_idx * 2
        closest_input = input_image_stack[n:n + 2]
        closest_input = np.transpose(closest_input, [1, 2, 0])
    else:
        raise ValueError(f'prev_selection_method "{args.prev_selection_method}" not implemented')

    if args.simulated_change_shift is not None:
        target_image = np.transpose(orig_target_image, [1, 2, 0])
        shift_mask = np.zeros_like(target_image)
        change_mask_bool = simulated_change_mask != 0.0
        shift_mask[change_mask_bool, :] = args.simulated_change_shift
        simulated_change_image = target_image + shift_mask

    prediction_diff = simulated_change_image - target_image_prediction
    previous_diff = simulated_change_image - closest_input

    prediction_diff = np.sqrt(np.sum(np.square(prediction_diff), axis=-1))
    previous_diff = np.sqrt(np.sum(np.square(previous_diff), axis=-1))

    y_mask = (simulated_change_mask != 0).astype(dtype=np.float32)

    return (prediction_diff, previous_diff, y_mask)


def sample_generator(ds, yield_all_pixels=False):
    def g():
        for img_sample in ds:
            prediction_diff, previous_diff, is_change_pixel = img_sample
            x_size, y_size = is_change_pixel.shape[0:2]
            if yield_all_pixels:
                for x in range(x_size):
                    for y in range(y_size):
                        datapoint = np.array(
                            [
                                prediction_diff[x, y],
                                previous_diff[x, y],
                                is_change_pixel[x, y]
                            ]
                        )
                        yield datapoint
            else:
                change_indices = np.where(is_change_pixel == 1.0)
                for x, y in zip(*change_indices):
                    datapoint = np.array(
                        [
                            prediction_diff[x, y],
                            previous_diff[x, y],
                            is_change_pixel[x, y]
                        ]
                    )
                    yield datapoint
                    #for every change pixel yield also one non change pixel
                    while True:
                        x_rand = random.randint(0, x_size - 1)
                        y_rand = random.randint(0, y_size - 1)
                        if is_change_pixel[x_rand, y_rand] != 1.0:
                            datapoint = np.array(
                                [
                                    prediction_diff[x_rand, y_rand],
                                    previous_diff[x_rand, y_rand],
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
            'prev_selection_method': args.prev_selection_method,
            'all_pixels': args.all_pixels,
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
                x['target_image_start_time'].numpy(),
                x['input_image_start_times'].numpy(),
            ),
        ds,
    )
    ds = filter(filter_samples, ds)
    image_samples = list(process_pool.imap_unordered(simulated_change_diff_samples, ds))
    num_imgs = len(image_samples)

    print(f'Num images {num_imgs}')

    arr = np.array(list(sample_generator(image_samples, yield_all_pixels=args.all_pixels)()))
    x_arr = arr[:, 0:2]
    y_arr = arr[:, 2]

    arr_pred_diff = x_arr[:, 0]
    arr_prev_diff = x_arr[:, 1]

    pred_roc = roc_curve(y_arr, arr_pred_diff)
    pred_auc = auc(pred_roc[0], pred_roc[1])
    prev_roc = roc_curve(y_arr, arr_prev_diff)
    prev_auc = auc(prev_roc[0], prev_roc[1])

    fig, axs = plt.subplots(1, 1)
    axs.set_box_aspect(1)

    RocCurveDisplay(fpr=pred_roc[0], tpr=pred_roc[1], estimator_name='Proposed Method', roc_auc=pred_auc).plot(ax=axs, color='dimgray', linestyle='-')
    RocCurveDisplay(fpr=prev_roc[0], tpr=prev_roc[1], estimator_name='Conventional Method', roc_auc=prev_auc).plot(ax=axs, color='dimgray', linestyle='--')

    if args.show_plot:
        axs.tick_params(labelsize=20)
        axs.xaxis.label.set_size(25)
        axs.yaxis.label.set_size(25)
        plt.legend(fontsize=25, loc='lower right')

        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(logs_path_obj / 'roc.png')


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
        '--all_pixels',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Compute the result for all possible pixels',
    )
    parser.add_argument(
        "--num_parallel",
        type=int,
        default=None,
        help="Parallel processing pool size",
    )
    parser.add_argument(
        '--prev_selection_method',
        type=str,
        default='1',
        choices=['1', '2'],
        help='Difference image computation method',
    )
    parser.add_argument(
        '--simulated_change_shift',
        type=float,
        default=None,
        help='Use shift simulated change method and add the argument value to the pixel values',
    )
    parser.add_argument(
        '--show_plot',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Open a matplotlib window with the plot',
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
