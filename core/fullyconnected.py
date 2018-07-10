import tensorflow as tf
from core.base_model import BaseModel


class FullyConnected(BaseModel):
    @staticmethod
    def build_model(inputs, fullyconnected_config):
        logits = tf.contrib.layers.fully_connected(
            inputs=inputs,
            num_outputs=fullyconnected_config['num_outputs'],
            activation_fn=fullyconnected_config['activation_fn'],
            normalizer_fn=fullyconnected_config['normalizer_fn'],
            normalizer_params=fullyconnected_config['normalizer_params'],
            weights_initializer=fullyconnected_config['weights_initializer'],
            weights_regularizer=fullyconnected_config['weights_regularizer'],
            biases_initializer=fullyconnected_config['biases_initializer'],
            biases_regularizer=fullyconnected_config['biases_regularizer'],
            reuse=fullyconnected_config['reuse'],
            variables_collections=fullyconnected_config['variables_collections'],
            outputs_collections=fullyconnected_config['outputs_collections'],
            trainable=fullyconnected_config['trainable'],
            scope=fullyconnected_config['scope']
        )
        return logits
