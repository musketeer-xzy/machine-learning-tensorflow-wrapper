import tensorflow as tf
from core.base_model import BaseModel


class CNN(BaseModel):
    @staticmethod
    def build_model(inputs, encoder_config):
        logits = [inputs]
        conv_list = []
        pool_list = []
        norm_list = []
        for i in range(encoder_config['layers_num']):
            with tf.variable_scope('conv' + str(i)) as scope:
                if encoder_config['conv_type'] == '1d':
                    conv = tf.layers.conv1d
                elif encoder_config['conv_type'] == '3d':
                    conv = tf.layers.conv3d
                else:
                    conv = tf.layers.conv2d
                conv_list.append(conv(
                    inputs=logits[-1],
                    filters=encoder_config['filters'][i],
                    kernel_size=encoder_config['kernel_size'][i],
                    strides=encoder_config['strides'][i],
                    padding=encoder_config['padding'][i],
                    data_format=encoder_config['data_format'][i],
                    dilation_rate=encoder_config['dilation_rate'][i],
                    activation=encoder_config['activation'][i],
                    use_bias=encoder_config['use_bias'][i],
                    kernel_initializer=encoder_config['kernel_initializer'][i],
                    bias_initializer=encoder_config['bias_initializer'][i],
                    kernel_regularizer=encoder_config['kernel_regularizer'][i],
                    bias_regularizer=encoder_config['bias_regularizer'][i],
                    activity_regularizer=encoder_config['activity_regularizer'][i],
                    kernel_constraint=encoder_config['kernel_constraint'][i],
                    bias_constraint=encoder_config['bias_constraint'][i],
                    trainable=encoder_config['trainable'][i],
                    name=scope.name,
                    reuse=encoder_config['reuse'][i]
                ))

            if encoder_config['pool_norm'][i] == 'pool':
                pool_list.append(tf.layers.max_pooling2d(
                    inputs=conv_list[-1],
                    pool_size=encoder_config['pool_size'][i],
                    strides=encoder_config['pool_strides'][i],
                    padding=encoder_config['pool_padding'][i],
                    data_format=encoder_config['pool_data_format'][i],
                    name='pool' + str(i)
                ))
                logits.append(pool_list[-1])
            elif encoder_config['pool_norm'][i] == 'norm':
                norm_list.append(tf.nn.lrn(
                    input=conv_list[-1],
                    depth_radius=encoder_config['norm_depth_radius'][i],
                    bias=encoder_config['norm_bias'][i],
                    alpha=encoder_config['norm_alpha'][i],
                    beta=encoder_config['norm_beta'][i],
                    name='norm' + str(i)
                ))
                logits.append(norm_list[-1])
            elif encoder_config['pool_norm'][i] == 'poolnorm':
                pool_list.append(tf.layers.max_pooling2d(
                    inputs=conv_list[-1],
                    pool_size=encoder_config['pool_size'][i],
                    strides=encoder_config['pool_strides'][i],
                    padding=encoder_config['pool_padding'][i],
                    data_format=encoder_config['pool_data_format'][i],
                    name='pool' + str(i)
                ))
                norm_list.append(tf.nn.lrn(
                    input=pool_list[-1],
                    depth_radius=encoder_config['norm_depth_radius'][i],
                    bias=encoder_config['norm_bias'][i],
                    alpha=encoder_config['norm_alpha'][i],
                    beta=encoder_config['norm_beta'][i],
                    name='norm' + str(i)
                ))
                logits.append(norm_list[-1])
            elif encoder_config['pool_norm'][i] == 'normpool':
                norm_list.append(tf.nn.lrn(
                    input=conv_list[-1],
                    depth_radius=encoder_config['norm_depth_radius'][i],
                    bias=encoder_config['norm_bias'][i],
                    alpha=encoder_config['norm_alpha'][i],
                    beta=encoder_config['norm_beta'][i],
                    name='norm' + str(i)
                ))
                pool_list.append(tf.layers.max_pooling2d(
                    inputs=norm_list[-1],
                    pool_size=encoder_config['pool_size'][i],
                    strides=encoder_config['pool_strides'][i],
                    padding=encoder_config['pool_padding'][i],
                    data_format=encoder_config['pool_data_format'][i],
                    name='pool' + str(i)
                ))
                logits.append(pool_list[-1])

        return logits[-1]

