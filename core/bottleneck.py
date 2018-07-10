import tensorflow as tf
from core.base_model import BaseModel


class BottleNeck(BaseModel):
    @staticmethod
    def build_model(inputs, bottleneck_config):
        logits = [inputs]
        for i in range(bottleneck_config['layers_num']):
            with tf.variable_scope('local' + str(i)) as scope:
                logits.append(tf.contrib.layers.fully_connected(
                    inputs=logits[-1],
                    num_outputs=bottleneck_config['num_outputs'][i],
                    activation_fn=bottleneck_config['activation_fn'][i],
                    normalizer_fn=bottleneck_config['normalizer_fn'][i],
                    normalizer_params=bottleneck_config['normalizer_params'][i],
                    weights_initializer=bottleneck_config['weights_initializer'][i],
                    weights_regularizer=bottleneck_config['weights_regularizer'][i],
                    biases_initializer=bottleneck_config['biases_initializer'][i],
                    biases_regularizer=bottleneck_config['biases_regularizer'][i],
                    reuse=bottleneck_config['reuse'][i],
                    variables_collections=bottleneck_config['variables_collections'][i],
                    outputs_collections=bottleneck_config['outputs_collections'][i],
                    trainable=bottleneck_config['trainable'][i],
                    scope=bottleneck_config['scope'][i]
                ))
            if bottleneck_config['dropout'][i]:
                logits.append(tf.nn.dropout(
                    logits[-1], keep_prob=bottleneck_config['keep_prob'][i], name=scope.name))

        return logits[-1]
