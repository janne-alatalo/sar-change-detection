import dataclasses
import typing

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from dataset_functions import ignore_weather_info, include_condition_data

def similarity_loss(input_image_stack, target_img, predicted, scaler):

    input_sar_imgs = input_image_stack[:, :, :, 0:-1]                           # bs, 512, 512, 8 (drop the dem raster)
    all_sar_imgs = tf.concat([input_sar_imgs, target_img], axis=-1)             # bs, 512, 512, 10
    s = tf.shape(all_sar_imgs)
    all_sar_imgs = tf.reshape(all_sar_imgs, [s[0], s[1], s[2], -1, 2])          # bs, 512, 512, 5, 2
    all_sar_imgs = tf.transpose(all_sar_imgs, perm=[3, 0, 1, 2, 4])             # 5, bs, 512, 512, 2
    img_diff = tf.reduce_mean(tf.square(all_sar_imgs - predicted), [2, 3, 4])   # 5, bs
    img_diff = tf.transpose(img_diff)                                           # bs, 5
    img_diff = img_diff * -1.0 # argmax to argmin hack
    img_one_hots = tf.nn.softmax(img_diff)

    # TODO: number of images is hardcoded
    true_one_hot = tf.one_hot(tf.ones(s[0], dtype=tf.dtypes.int32) * 4, 5)

    return scaler * tf.reduce_mean(tf.losses.categorical_crossentropy(true_one_hot, img_one_hots))


# Loss function that amplifies the simulation error over forested areas and
# decreases the error where there are no forest, like lakes. The
# non_forest_scaler tells how much to decrease the non forest area error.
def forestmask_loss(target_img, predicted_img, forestmask, non_forest_scaler=0.1):
    scaler_mask = tf.ones_like(forestmask)
    forestmask_bool = tf.not_equal(forestmask, tf.constant(1.0))
    update_idxs = tf.where(forestmask_bool)
    update_vals = tf.ones(tf.shape(update_idxs)[0]) * non_forest_scaler
    scaler_mask = tf.tensor_scatter_nd_update(scaler_mask, update_idxs, update_vals)
    scaler_mask = tf.stack([
        scaler_mask,
        scaler_mask,
    ], axis=-1)

    loss = tf.reduce_mean(tf.square(target_img - predicted_img) * scaler_mask)
    return loss


# Some code and inspiration from here:
# https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
class SARWeatherUNet(tf.keras.Model):

    def __init__(self, model_params, drop_path_rate: float):
        super().__init__()

        total_res_modules = 9 + 8
        drop_path_rates = list(1.0 - np.linspace(0.0, drop_path_rate, total_res_modules))

        self.downsampler = Downsampler(model_params, drop_path_rates[0:9])
        self.upsampler = Upsampler(model_params, drop_path_rates[9:])
        self.model_params = model_params

    def call(self, samples, training=None):
        input = samples['input_image_stack']
        latent_metadata = samples['latent_metadata']
        if self.model_params.no_weather_data:
            (
                input_platform_headings,
                input_incidence_angles,
                input_mission_ids,
                target_platform_heading,
                target_incidence_angle,
                target_mission_id,
            ) = ignore_weather_info(latent_metadata)
            target_data = tf.stack(
                [
                    target_platform_heading,
                    target_incidence_angle,
                    target_mission_id,
                ],
                axis=-1,
            )
            latent_metadata = tf.concat(
                [
                    input_platform_headings,
                    input_incidence_angles,
                    input_mission_ids,
                    target_data,
                ],
                axis=-1
            )
        elif self.model_params.ablation_study is not None:
            if self.model_params.ablation_study == 'temperature':
                (
                    #input_temperatures,
                    input_snow_depths,
                    input_platform_headings,
                    input_incidence_angles,
                    input_precipitation,
                    input_mission_ids,
                    #target_temperature,
                    target_snow_depth,
                    target_platform_heading,
                    target_incidence_angle,
                    target_mission_id,
                    target_precipitation,
                ) = include_condition_data(latent_metadata, temperature=False)
                latent_metadata = tf.concat(
                    [
                        #input_temperatures,
                        input_snow_depths,
                        input_platform_headings,
                        input_incidence_angles,
                        input_precipitation,
                        input_mission_ids,
                        tf.stack(
                            [
                                #target_temperature,
                                target_snow_depth,
                                target_platform_heading,
                                target_incidence_angle,
                                target_mission_id,
                            ],
                            axis=-1,
                        ),
                        target_precipitation,
                    ],
                    axis=-1,
                )
            elif self.model_params.ablation_study == 'snow_depth':
                (
                    input_temperatures,
                    #input_snow_depths,
                    input_platform_headings,
                    input_incidence_angles,
                    input_precipitation,
                    input_mission_ids,
                    target_temperature,
                    #target_snow_depth,
                    target_platform_heading,
                    target_incidence_angle,
                    target_mission_id,
                    target_precipitation,
                ) = include_condition_data(latent_metadata, snow_depth=False)
                latent_metadata = tf.concat(
                    [
                        input_temperatures,
                        #input_snow_depths,
                        input_platform_headings,
                        input_incidence_angles,
                        input_precipitation,
                        input_mission_ids,
                        tf.stack(
                            [
                                target_temperature,
                                #target_snow_depth,
                                target_platform_heading,
                                target_incidence_angle,
                                target_mission_id,
                            ],
                            axis=-1,
                        ),
                        target_precipitation,
                    ],
                    axis=-1,
                )
            elif self.model_params.ablation_study == 'platform_heading':
                (
                    input_temperatures,
                    input_snow_depths,
                    #input_platform_headings,
                    input_incidence_angles,
                    input_precipitation,
                    input_mission_ids,
                    target_temperature,
                    target_snow_depth,
                    #target_platform_heading,
                    target_incidence_angle,
                    target_mission_id,
                    target_precipitation,
                ) = include_condition_data(latent_metadata, platform_heading=False)
                latent_metadata = tf.concat(
                    [
                        input_temperatures,
                        input_snow_depths,
                        #input_platform_headings,
                        input_incidence_angles,
                        input_precipitation,
                        input_mission_ids,
                        tf.stack(
                            [
                                target_temperature,
                                target_snow_depth,
                                #target_platform_heading,
                                target_incidence_angle,
                                target_mission_id,
                            ],
                            axis=-1,
                        ),
                        target_precipitation,
                    ],
                    axis=-1,
                )
            elif self.model_params.ablation_study == 'incidence_angle':
                (
                    input_temperatures,
                    input_snow_depths,
                    input_platform_headings,
                    #input_incidence_angles,
                    input_precipitation,
                    input_mission_ids,
                    target_temperature,
                    target_snow_depth,
                    target_platform_heading,
                    #target_incidence_angle,
                    target_mission_id,
                    target_precipitation,
                ) = include_condition_data(latent_metadata, incidence_angle=False)
                latent_metadata = tf.concat(
                    [
                        input_temperatures,
                        input_snow_depths,
                        input_platform_headings,
                        #input_incidence_angles,
                        input_precipitation,
                        input_mission_ids,
                        tf.stack(
                            [
                                target_temperature,
                                target_snow_depth,
                                target_platform_heading,
                                #target_incidence_angle,
                                target_mission_id,
                            ],
                            axis=-1,
                        ),
                        target_precipitation,
                    ],
                    axis=-1,
                )
            elif self.model_params.ablation_study == 'precipitation':
                (
                    input_temperatures,
                    input_snow_depths,
                    input_platform_headings,
                    input_incidence_angles,
                    #input_precipitation,
                    input_mission_ids,
                    target_temperature,
                    target_snow_depth,
                    target_platform_heading,
                    target_incidence_angle,
                    target_mission_id,
                    #target_precipitation,
                ) = include_condition_data(latent_metadata, precipitation=False)
                latent_metadata = tf.concat(
                    [
                        input_temperatures,
                        input_snow_depths,
                        input_platform_headings,
                        input_incidence_angles,
                        #input_precipitation,
                        input_mission_ids,
                        tf.stack(
                            [
                                target_temperature,
                                target_snow_depth,
                                target_platform_heading,
                                target_incidence_angle,
                                target_mission_id,
                            ],
                            axis=-1,
                        ),
                        #target_precipitation,
                    ],
                    axis=-1,
                )
            elif self.model_params.ablation_study == 'mission_id':
                (
                    input_temperatures,
                    input_snow_depths,
                    input_platform_headings,
                    input_incidence_angles,
                    input_precipitation,
                    #input_mission_ids,
                    target_temperature,
                    target_snow_depth,
                    target_platform_heading,
                    target_incidence_angle,
                    #target_mission_id,
                    target_precipitation,
                ) = include_condition_data(latent_metadata, mission_id=False)
                latent_metadata = tf.concat(
                    [
                        input_temperatures,
                        input_snow_depths,
                        input_platform_headings,
                        input_incidence_angles,
                        input_precipitation,
                        #input_mission_ids,
                        tf.stack(
                            [
                                target_temperature,
                                target_snow_depth,
                                target_platform_heading,
                                target_incidence_angle,
                                #target_mission_id,
                            ],
                            axis=-1,
                        ),
                        target_precipitation,
                    ],
                    axis=-1,
                )

        x, skips = self.downsampler(input, training=training)
        lmd_shape = latent_metadata.shape
        # bs, len(one_sample_latent_metdata) -> bs, 1, 1, len(one_sample_latent_metdata)
        latent_metadata = tf.reshape(latent_metadata, [-1, 1, 1, lmd_shape[1]])
        x = tf.concat([x, latent_metadata], axis=-1) # bs, 1, 1, 512 + len(one_sample_latent_metdata)
        x = self.upsampler(x, skips, training=training) # bs, 512, 512, 2

        # For mixed precision, make sure that the model outputs float32
        # If mixed precision is off, this is a no-op
        x = tf.keras.layers.Activation('linear', dtype='float32')(x)

        return x


@dataclasses.dataclass()
class ModelParams():
    downsampler_layers: list[tuple[str, int, int, int]]
    upsampler_layers: list[tuple[str, int, int, int]]
    frontend_params: typing.Union[list[tuple[int, int]], None]
    backend_params: typing.Union[list[tuple[int, int]], None]
    unet_skips: str
    resblock_bottleneck_multiplier: int
    no_weather_data: bool
    ablation_study: typing.Union[str, None]
    dropout: float = 0.0

    def asdict(self):
        return dataclasses.asdict(self)

class ResBlock(tf.keras.layers.Layer):

    def __init__(
            self,
            filters: int,
            is_identity: bool = True,
            drop_path_rate: float = 0.0,
            kernel_size: tuple[int, int] = (7, 7),
            bottleneck_multiplier: int = 4,
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)

        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            activation=None,
            padding='same',
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            bottleneck_multiplier * filters,
            kernel_size=(1, 1),
            activation='gelu',
            padding='valid',
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            padding='valid',
        )
        self.conv4 = None
        if not is_identity:
            self.conv4 = tf.keras.layers.Conv2D(
                filters,
                kernel_size=(1, 1),
                padding='valid',
            )
        self.stochastic_depth = tfa.layers.StochasticDepth(drop_path_rate)

    def call(self, input, training=None):
        x = self.conv1(input, training=training)
        x = self.bn(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        if self.conv4 is not None:
            input = self.conv4(input)
        return self.stochastic_depth([x, input], training=training)


class ResModule(tf.keras.layers.Layer):

    def __init__(
            self,
            filters: int,
            num_blocks: int = 1,
            is_identity: bool = True,
            drop_path_rate: float = 0.0,
            kernel_size: tuple[int, int] = (7, 7),
            bottleneck_multiplier: int = 4,
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)

        self.blocks = []
        self.blocks.append(
            ResBlock(
                filters,
                is_identity=is_identity,
                drop_path_rate=drop_path_rate,
                kernel_size=kernel_size,
                bottleneck_multiplier=bottleneck_multiplier,
            )
        )
        if num_blocks > 1:
            for _ in range(num_blocks - 1):
                self.blocks.append(
                    ResBlock(
                        filters,
                        is_identity=True,
                        drop_path_rate=drop_path_rate,
                        kernel_size=kernel_size,
                        bottleneck_multiplier=bottleneck_multiplier,
                    )
                )

    def call(self, input, training=None):
        y = input
        for b in self.blocks:
            y = b(y, training=training)

        return y


class Downsampler(tf.keras.Model):

    def __init__(self, model_params: ModelParams, drop_path_rates: list[float]):
        super().__init__()

        frontend_layers = []
        if model_params.frontend_params is not None:
            for num_filters, kernel_size in model_params.frontend_params:
                frontend_layers.append(
                    tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=(1, 1), padding='same'),
                )
            self.frontend = tf.keras.Sequential(frontend_layers)
        else:
            self.frontend = None

        self.model_params = model_params
        ls = model_params.downsampler_layers

        # input shape is bs, 512, 512, 5
        self.blocks = [
            self.downsample_block(*ls[0], is_identity=False, drop_path_rate=drop_path_rates[0]),
            self.downsample_block(*ls[1], is_identity=(ls[0][2] == ls[1][2]), drop_path_rate=drop_path_rates[1]),
            self.downsample_block(*ls[2], is_identity=(ls[1][2] == ls[2][2]), drop_path_rate=drop_path_rates[2]),
            self.downsample_block(*ls[3], is_identity=(ls[2][2] == ls[3][2]), drop_path_rate=drop_path_rates[3]),
            self.downsample_block(*ls[4], is_identity=(ls[3][2] == ls[4][2]), drop_path_rate=drop_path_rates[4]),
            self.downsample_block(*ls[5], is_identity=(ls[4][2] == ls[5][2]), drop_path_rate=drop_path_rates[5]),
            self.downsample_block(*ls[6], is_identity=(ls[5][2] == ls[6][2]), drop_path_rate=drop_path_rates[6]),
            self.downsample_block(*ls[7], is_identity=(ls[6][2] == ls[7][2]), drop_path_rate=drop_path_rates[7]),
            self.downsample_block(*ls[8], is_identity=(ls[7][2] == ls[8][2]), drop_path_rate=drop_path_rates[8]),
        ]

    def downsample_block(self, layer_type, num_blocks, num_filters, kernel_size, is_identity=True, drop_path_rate: float = 0.0):
        if layer_type == 'r':
            b = tf.keras.Sequential(
                [
                    ResModule(
                        num_filters,
                        num_blocks,
                        kernel_size=(kernel_size, kernel_size),
                        is_identity=is_identity,
                        drop_path_rate=drop_path_rate,
                        bottleneck_multiplier=self.model_params.resblock_bottleneck_multiplier,
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(
                        num_filters,
                        kernel_size=(2, 2),
                        strides=(2, 2),
                    ),
                ]
            )
        elif layer_type == 't':
            if num_blocks != 1:
                raise NotImplementedError('Layer type "t" num_blocks must be 1')
            b = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        num_filters,
                        kernel_size,
                        strides=(2, 2),
                        padding='same',
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                ]
            )

        else:
            raise NotImplementedError(f'Layer type "{layer_type}" not implemented')

        return b

    def call(self, inputs, training=None):
        # Currently hadrcoded, 9 comes from num_pre_images * num_channels + dem_raster
        tf.ensure_shape(inputs, [None, 512, 512, 9])
        if self.frontend is not None:
            inputs = self.frontend(inputs, training=training)
        skips = []
        x = inputs

        for b in self.blocks:
            x = b(x, training=training)
            skips.append(x)

        return x, skips[:-1]

class Upsampler(tf.keras.Model):

    def __init__(self, model_params: ModelParams, drop_path_rates: list[float]):
        super().__init__()

        self.model_params = model_params
        ls = model_params.upsampler_layers

        self.blocks = [
            self.upsample_block(*ls[0], drop_path_rate=drop_path_rates[0], dropout=model_params.dropout),
            self.upsample_block(*ls[1], drop_path_rate=drop_path_rates[1], dropout=model_params.dropout),
            self.upsample_block(*ls[2], drop_path_rate=drop_path_rates[2], dropout=model_params.dropout),
            self.upsample_block(*ls[3], drop_path_rate=drop_path_rates[3]),
            self.upsample_block(*ls[4], drop_path_rate=drop_path_rates[4]),
            self.upsample_block(*ls[5], drop_path_rate=drop_path_rates[5]),
            self.upsample_block(*ls[6], drop_path_rate=drop_path_rates[6]),
            self.upsample_block(*ls[7], drop_path_rate=drop_path_rates[7]),
        ]


        backend_layers = []
        if model_params.backend_params is not None:
            for num_filters, kernel_size in model_params.backend_params:
                backend_layers.append(
                    tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=(1, 1), padding='same'),
                )
            self.backend = tf.keras.Sequential(backend_layers)
        else:
            self.backend = None

        self.last_layer = tf.keras.layers.Conv2DTranspose(2, 4, strides=(2, 2), padding='same', activation='tanh')
        self.skips_enabled = model_params.unet_skips[::-1]

    def upsample_block(self, layer_type, num_blocks, num_filters, kernel_size, is_identity=False, drop_path_rate: float = 0.0, dropout=None):
        if layer_type == 'r':
            b = tf.keras.Sequential(
                [
                    ResModule(
                        num_filters,
                        num_blocks,
                        kernel_size=(kernel_size, kernel_size),
                        is_identity=is_identity,
                        drop_path_rate=drop_path_rate,
                        bottleneck_multiplier=self.model_params.resblock_bottleneck_multiplier,
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2DTranspose(
                        num_filters,
                        kernel_size=(2, 2),
                        strides=(2, 2),
                        padding='same',
                    ),
                ]
            )
        elif layer_type == 't':
            if num_blocks != 1:
                raise NotImplementedError('Layer type "t" num_blocks must be 1')
            b = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2DTranspose(
                        num_filters,
                        kernel_size,
                        strides=(2, 2),
                        padding='same',
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),
                ]
            )

        else:
            raise NotImplementedError(f'Layer type "{layer_type}" not implemented')

        if dropout:
            b.add(tf.keras.layers.Dropout(dropout))

        return b

    def call(self, input, skips, training=None):
        x = input

        for i, (b, s) in enumerate(zip(self.blocks, skips[::-1])):
            x = b(x, training=training)
            if i < 7 and self.skips_enabled[i] == '1':
                x = tf.concat([x, s], axis=-1)

        if self.backend is not None:
            x = self.backend(x)
        x = self.last_layer(x, training=training) # bs, 512, 512, 2

        return x
