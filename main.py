import argparse
import glob
import json
import datetime
import os
import pathlib
import multiprocessing
import re
import signal

import configargparse
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import git

from model import SARWeatherUNet, similarity_loss, ModelParams, forestmask_loss
from dataset import get_dataset, denormalize_imgs


cpu_count = multiprocessing.cpu_count()

EARLY_STOP = False


class CustomTensorBoardCallback(tf.keras.callbacks.TensorBoard):

    def __init__(self, test_samples, ds_stats, update_freq='epoch', *args, **kwargs):
        super().__init__(*args, update_freq=update_freq, **kwargs)
        self.test_samples = test_samples
        self.inputs, self.targets = next(iter(test_samples.batch(len(list(test_samples)))))
        self.input_imgs = self.inputs['input_image_stack']
        self.ds_stats = ds_stats

        denorm_imgs = denormalize_imgs(self.targets, self.ds_stats)
        self.img_maxs = np.nanmax(np.reshape(denorm_imgs, [denorm_imgs.shape[0], -1]), axis=-1)
        self.img_mins = np.nanmin(np.reshape(denorm_imgs, [denorm_imgs.shape[0], -1]), axis=-1)

        per_batch_logging = True if update_freq != 'epoch' else False

        self.imgs = self.transform_to_imgs(denorm_imgs)
        self.per_batch_logging = per_batch_logging

    def transform_to_imgs(self, imgs):
        imgs = np.transpose(imgs, [1, 2, 3, 0])
        imgs = (imgs - self.img_mins) / (self.img_maxs - self.img_mins)
        chan3 = np.divide(imgs[:, :, 1, :], imgs[:, :, 0, :], where=imgs[:, :, 0, :] != 0.0)
        chan3[imgs[:, :, 0, :] == 0.0] = 0.0
        s = chan3.shape
        chan3 = np.reshape(chan3, [s[0], s[1], 1, s[2]])
        imgs = np.concatenate([imgs, chan3], axis=2)
        imgs = np.transpose(imgs, [3, 0, 1, 2])

        return imgs

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        num_test_samples = len(self.input_imgs)
        outputs = self.model(self.inputs).numpy()
        with self._val_writer.as_default():
            with tf.summary.record_if(True):
                if epoch == 0:
                    tf.summary.image('test_image', self.imgs, max_outputs=num_test_samples, step=0)
                denorm_imgs = denormalize_imgs(outputs, self.ds_stats)
                imgs = self.transform_to_imgs(denorm_imgs)
                tf.summary.image('test_image', imgs, max_outputs=num_test_samples, step=epoch + 1)

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        if not self.per_batch_logging:
            return
        if logs is None or 'loss' not in logs:
            return
        with self._train_writer.as_default():
            tf.summary.scalar('batch_loss_custom_logged', logs['loss'], step=self._global_train_batch)
        if EARLY_STOP:
            self.model.stop_training = True

    def on_val_batch_end(self, _batch, _logs=None):
        super().on_val_batch_end(batch, logs)
        if EARLY_STOP:
            self.model.stop_training = True


def interrupt_handler(signum, frame):
    global EARLY_STOP
    EARLY_STOP = True


def main(args):

    if not args.no_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    if repo.is_dirty() and not args.ignore_dirty_repo:
        print('Repository is dirty commit changes or pass --ignore_dirty_repo!')
        exit(1)

    model_params = ModelParams(
        downsampler_layers=args.downsampler_layers,
        upsampler_layers=args.upsampler_layers,
        frontend_params=args.frontend_params,
        backend_params=args.backend_params,
        unet_skips=args.unet_skips,
        resblock_bottleneck_multiplier=args.resblock_bottleneck_multiplier,
        dropout=args.dropout,
    )

    datenow_tag = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    tag = datenow_tag
    logs_path = os.path.join('./logs', tag)
    logs_path_obj = pathlib.Path(logs_path)
    i = 1
    while True:
        try:
            logs_path_obj.mkdir(parents=True)
            break
        except FileExistsError:
            tag = f'{datenow_tag}_{i}'
            logs_path = os.path.join('./logs', tag)
            logs_path_obj = pathlib.Path(logs_path)
            i += 1

    train_files = [f for train_glob in args.train_data for f in glob.glob(train_glob)]
    val_files = [f for val_glob in args.val_data for f in glob.glob(val_glob)]

    num_parallel_reads = args.num_parallel_reads if args.num_parallel_reads is not None else cpu_count
    train_files.sort()
    val_files.sort()

    train_args = json.dumps({
        'model_params': model_params.asdict(),
        'resume_from_checkpoint': args.resume_from_checkpoint,
        'keep_optimizer_state': args.keep_optimizer_state,
        'batch_size': args.batch_size,
        'mixed_precision': not args.no_mixed_precision,
        'adamw_weight_decay': args.adamw_weight_decay,
        'drop_path_rate': args.drop_path_rate,
        'learning_rate': args.learning_rate,
        'optimizer_epsilon': args.optimizer_epsilon,
        'repo_commit': sha,
        'repo_is_dirty': repo.is_dirty(),
        'logs_path': os.path.abspath(logs_path),
        'run_eagerly': args.run_eagerly,
        'num_parallel_reads': num_parallel_reads,
        'similarity_loss_scaler': args.similarity_loss_scaler,
        'non_forest_scaler': args.forestmask_non_forest_scaler,
        'train_data': train_files,
        'val_data': val_files,
        'tb_test_dataset': args.tb_test_dataset,
    }, indent=4)
    train_filepath = logs_path_obj / 'train_args.json'
    with train_filepath.open('w', encoding='utf-8') as f:
        f.write(train_args)

    if repo.is_dirty():
        patch = repo.git.diff()
        patch_filepath = logs_path_obj / 'patch.diff'
        with patch_filepath.open('w', encoding='utf-8') as f:
            f.write(patch)
    print('==== TRAIN arguments ====')
    print(train_args)
    print('=========================')

    with open(args.dataset_stats, 'r') as f:
        ds_stats = json.load(f)

    train_dataset_batch = get_dataset(
        train_files,
        compression_type=args.tfrecord_compression,
        num_parallel_reads=num_parallel_reads,
        batch_size=args.batch_size,
    )
    val_dataset_batch = get_dataset(
        val_files,
        compression_type=args.tfrecord_compression,
        num_parallel_reads=num_parallel_reads,
        batch_size=args.batch_size,
    )

    if args.tb_test_dataset is not None:
        tb_test_images = get_dataset(
            args.tb_test_dataset,
            compression_type=args.tfrecord_compression,
            num_parallel_reads=num_parallel_reads,
            batch_size=10,
        )
        tb_test_images = tb_test_images.unbatch().take(12)
    else:
        tb_test_images = val_dataset_batch.unbatch().take(12)

    if args.prefetch != 0:
        prefetch_num = args.prefetch if args.prefetch is not None else tf.data.AUTOTUNE
        train_dataset_batch = train_dataset_batch.prefetch(prefetch_num)
        val_dataset_batch = val_dataset_batch.prefetch(prefetch_num)

    profile_batch = args.profile_batch if args.profile_batch is not None else 0

    tb_write_frequency = args.tb_write_frequency
    if tb_write_frequency.isnumeric():
        tb_write_frequency = int(tb_write_frequency)
    tb_callback = CustomTensorBoardCallback(
        tb_test_images,
        ds_stats,
        log_dir=logs_path,
        profile_batch=profile_batch,
        histogram_freq=1,
        update_freq=tb_write_frequency,
    )
    callbacks = [
        tb_callback,
    ]
    if not args.no_checkpoints:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=logs_path_obj / 'checkpoints/{epoch:02d}/',
        ))

    if args.resume_from_checkpoint:
        if args.keep_optimizer_state:
            model = tf.keras.models.load_model(args.resume_from_checkpoint, compile=True)
            model.run_eagerly = args.run_eagerly
        else:
            model = tf.keras.models.load_model(args.resume_from_checkpoint, compile=False)
            adamw_optimizer = tfa.optimizers.AdamW(
                learning_rate=args.learning_rate,
                weight_decay=args.adamw_weight_decay,
                epsilon=args.optimizer_epsilon,
            )

            model.compile(
                optimizer=adamw_optimizer,
                loss=None,
                metrics=[
                    'mse',
                ],
                run_eagerly=args.run_eagerly,
            )

        (
            input_image_stack,
            latent_metadata,
            target_image,
            forestmask,
        ) = model.inputs

        output = model.outputs[0]

    else:
        input_image_stack = tf.keras.Input(shape=(512, 512, 9))
        latent_metadata = tf.keras.Input(shape=(45,))
        target_image = tf.keras.Input(shape=(512, 512, 2))
        forestmask = tf.keras.Input(shape=(512, 512))

        _model = SARWeatherUNet(model_params, args.drop_path_rate)

        output = _model({
            'input_image_stack': input_image_stack,
            'latent_metadata': latent_metadata,
            'target_image': target_image,
        })

        model = tf.keras.Model(
            inputs={
                'input_image_stack': input_image_stack,
                'latent_metadata': latent_metadata,
                'target_image': target_image,
                'forestmask': forestmask,
            },
            outputs=output,
        )

        if args.similarity_loss_scaler != 0.0:
            sim_loss = similarity_loss(
                input_image_stack,
                target_image,
                output,
                args.similarity_loss_scaler,
            )
            model.add_loss(sim_loss)
            model.add_metric(sim_loss, 'similarity_loss')

        fm_loss = forestmask_loss(
            target_image,
            output,
            forestmask,
            non_forest_scaler=args.forestmask_non_forest_scaler,
        )
        model.add_loss(fm_loss)
        model.add_metric(fm_loss, 'forestmask_loss')

        if args.forestmask_non_forest_scaler == 0.0:
            fm_mse = fm_loss
        else:
            fm_mse = forestmask_loss(
                target_image,
                output,
                forestmask,
                non_forest_scaler=0.0,
            )
        model.add_metric(fm_mse, 'forestmask_mse')

        adamw_optimizer = tfa.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.adamw_weight_decay,
            epsilon=args.optimizer_epsilon,
        )

        model.compile(
            optimizer=adamw_optimizer,
            loss=None,
            metrics=[
                'mse',
            ],
            run_eagerly=args.run_eagerly,
        )

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, interrupt_handler)
    signal.signal(signal.SIGINT, interrupt_handler)

    model.fit(
        train_dataset_batch,
        validation_data=val_dataset_batch,
        callbacks=callbacks,
        epochs=args.epochs,
    )

    signal.signal(signal.SIGTERM, original_sigint_handler)
    signal.signal(signal.SIGINT, original_sigterm_handler)

    if EARLY_STOP or args.no_checkpoints:
        model.save(logs_path_obj / 'checkpoints/final/')


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

    def sampler_layer_params(pat):
        def f(arg_val) -> list[tuple[int, int]]:
            if not pat.match(arg_val):
                raise argparse.ArgumentTypeError
            params = []
            for lp in arg_val.split(','):
                if len(lp) == 0:
                    break
                t = lp[0]
                m = re.findall(r'\d+', lp)
                if len(m) != 3:
                    raise argparse.ArgumentTypeError
                params.append(tuple([t, int(m[0]), int(m[1]), int(m[2])]))
            return params
        return f

    def layer_params(pat):
        def f(arg_val) -> list[tuple[int, int]]:
            if not pat.match(arg_val):
                raise argparse.ArgumentTypeError
            params = []
            for lp in arg_val.split(','):
                m = re.findall(r'\d+', lp)
                if len(m) == 1:
                    params.append((int(m[0]), 1))
                else:
                    params.append(tuple([int(m[0]), int(m[1])]))
            return params
        return f

    def regex_str(pat):
        def f(arg_val) -> str:
            if not pat.match(arg_val):
                raise argparse.ArgumentTypeError
            return arg_val

        return f


    parser = configargparse.ArgumentParser()

    parser.add_argument(
        '--train_data',
        env_var='TRAIN_DATA',
        type=str,
        nargs='+',
        action='extend',
        required=True,
        help='Training dataset glob',
    )
    parser.add_argument(
        '--val_data',
        env_var='VAL_DATA',
        type=str,
        nargs='+',
        action='extend',
        required=True,
        help='Validation dataset glob',
    )
    parser.add_argument(
        '--tb_test_dataset',
        env_var='TB_TEST_DATASET',
        type=str,
        default=None,
        help='Example validation images that are used for TensorBoard examples',
    )
    parser.add_argument(
        '--dataset_stats',
        env_var='DATASET_STATS',
        type=str,
        default='stats.json',
        help='Dataset statistics that are used for normalization',
    )
    parser.add_argument(
        '--batch_size',
        env_var='BATCH_SIZE',
        type=int,
        default=100,
        help='Batch size',
    )
    parser.add_argument(
        '--epochs',
        env_var='EPOCHS',
        type=int,
        default=5,
        help='Number of epochs',
    )
    parser.add_argument(
        '--similarity_loss_scaler',
        env_var='SIMILARITY_LOSS_SCALER',
        type=float,
        default=0.0,
        help='Scaler for the similarity metric loss',
    )
    parser.add_argument(
        '--forestmask_non_forest_scaler',
        env_var='FORESTMASK_NON_FOREST_SCALER',
        type=float,
        default=1.0,
        help='Scale non forest area error with this scalar (when set to 1.0, the loss is MSE)',
    )
    parser.add_argument(
        '--drop_path_rate',
        env_var='DROP_PATH_RATE',
        type=float,
        default=0.0,
        help='Stochastic depth drop path rate',
    )
    parser.add_argument(
        '--dropout',
        env_var='DROPOUT',
        type=float,
        default=None,
        help='Dropout rate in Upsample layers.',
    )
    parser.add_argument(
        '--adamw_weight_decay',
        env_var='ADAMW_WEIGHT_DECAY',
        type=float,
        default=0.005,
        help='AdamW weight decay parameter.',
    )
    parser.add_argument(
        '--unet_skips',
        env_var='UNET_SKIPS',
        type=regex_str(re.compile(r'(:?0|1){8}')),
        default='11111111',
        help='Enable/Disable skip connections',
    )
    parser.add_argument(
        '--downsampler_layers',
        env_var='DOWNSAMPLER_LAYERS',
        type=sampler_layer_params(re.compile(r'((r|t)\(\d+\*\d+\*\d+\)(,|$)){9}')),
        default='t(1*64*4),t(1*128*4),t(1*256*4),t(1*512*4),t(1*512*4),t(1*512*4),t(1*512*4),t(1*512*4),t(1*512*2),',
        help='Downsampler layer configuration in the form of x(n*f*k) where x is the layer type, n is the number of blocks, f is the number of filters and k is the kernel size',
    )
    parser.add_argument(
        '--upsampler_layers',
        env_var='UPSAMPLER_LAYERS',
        type=sampler_layer_params(re.compile(r'((r|t)\(\d+\*\d+\*\d+\)(,|$)){8}')),
        default='t(1*512*4),t(1*512*4),t(1*512*4),t(1*512*4),t(1*512*4),t(1*256*4),t(1*128*4),t(1*64*4),',
        help='Upsampler layer configuration in the form of x(n*f*k) where x is the layer type, n is the number of blocks, f is the number of filters and k is the kernel size',
    )
    parser.add_argument(
        '--frontend_params',
        env_var='FRONTEND_PARAMS',
        type=layer_params(re.compile(r'(\(\d+\*\d+\)(,|$))+')),
        default=None,
        help='Frontend params in form (num_filters*kernel_size)',
    )
    parser.add_argument(
        '--backend_params',
        env_var='BACKEND_PARAMS',
        type=layer_params(re.compile(r'(\(\d+\*\d+\)(,|$))+')),
        default=None,
        help='Backend params in form (num_filters*kernel_size)',
    )
    parser.add_argument(
        '--learning_rate',
        env_var='LEARNING_RATE',
        default=0.001,
        type=float,
        help='Model training optimizer learning rate',
    )
    parser.add_argument(
        '--optimizer_epsilon',
        env_var='OPTIMIZER_EPSILON',
        default=1e-07,
        type=float,
        help='Optimizer epsilon value',
    )
    parser.add_argument(
        '--resblock_bottleneck_multiplier',
        env_var='RESBLOCK_BOTTLENECK_MULTIPLIER',
        type=int,
        default=4,
        help='The second layer in resblock is first_layer_filter_size * resblock_bottleneck_multiplier',
    )
    parser.add_argument(
        '--prefetch',
        env_var='PREFETCH',
        type=int,
        default=None,
        help='Prefetch batches in dataset pipeline (default is to autotune this)',
    )
    parser.add_argument(
        '--no_checkpoints',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Save checkpoints on each epoch',
    )
    parser.add_argument(
        '--run_eagerly',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Run the model in eager mode (useful for debugging)',
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
        '--no_mixed_precision',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Disable mixed precision computation',
    )
    parser.add_argument(
        '--profile_batch',
        type=str,
        default=None,
        help='Enable profiler and profile batch. (example --profile_batch "5,55" profiles batches 5-55)',
    )
    parser.add_argument(
        '-r', '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Resume trining from a checkpoint',
    )
    parser.add_argument(
        '--keep_optimizer_state',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='When resuming training from checkpoint, keep the optimizer state (optimizer parameter changes are ignored)',
    )
    parser.add_argument(
        "--tfrecord_compression",
        default="GZIP",
        choices=["GZIP", "ZLIB", ""],
        help="TFRecord compression type (set to \"\" to use no compression)",
    )
    parser.add_argument(
        "--num_parallel_reads",
        default=None,
        type=int,
        help="How many parallel reads for dataset tfrecord input pipeline",
    )
    parser.add_argument(
        "--tb_write_frequency",
        default='epoch',
        type=str,
        help="Can be 'epoch' or integer. When integer, the logs are written at that batch frequency",
    )

    args = parser.parse_args()

    main(args)
