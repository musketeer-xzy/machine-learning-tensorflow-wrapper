from configparser import ConfigParser
import os
import tensorflow as tf


def ctc_accuracy(logits, labels, parameters):
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(
        logits, parameters['seq_len'], merge_repeated=parameters['merge_repeated'])
    return tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))


DTYPES = {'float32': tf.float32}
ACTIVATIONS = {'relu': tf.nn.relu}
INITIALIZERS = {
    'zeros': lambda parameters: tf.zeros_initializer(),
    'xavier': lambda parameters: tf.contrib.layers.xavier_initializer(),
    'truncated_normal': lambda parameters: tf.truncated_normal_initializer(stddev=parameters['stddev']),
    'constant': lambda parameters: tf.constant_initializer(value=parameters['value'])
}

LOSSES = {'cross_entropy': lambda labels, logits, parameters: tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=labels, logits=logits, dim=parameters['dim'], name=parameters['name']),
          'ctc': lambda labels, logits, parameters: tf.nn.ctc_loss(
              labels=labels, inputs=logits, sequence_length=parameters['sequence_length'])}

OPTIMIZERS = {
    'GradientDescent':
        lambda loss, learning_rate, global_step, parameters=None: tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate, use_locking=parameters['use_locking'], name=parameters['name']).minimize(
            loss, global_step=global_step),
    'Momentum':
        lambda loss, learning_rate, global_step, parameters=None: tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=parameters['momentum']).minimize(loss, global_step=global_step),
    'Adam':
        lambda loss, learning_rate, global_step, parameters=None: tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=parameters['beta1'], beta2=parameters['beta2'],
            epsilon=parameters['epsilon'], use_locking=parameters['use_locking'], name=parameters['name']).minimize(
            loss, global_step=global_step)}

ACCURACYS = {'argmax': lambda labels, logits, parameters: tf.reduce_mean(tf.cast(tf.equal(tf.argmax(
    labels, parameters['axis']), tf.argmax(logits, parameters['axis'])), tf.float32)), 'ctc': ctc_accuracy}


class Config:
    
    def __init__(self, conf_name):
        conf_file_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'configuration', conf_name)
        self.__config = ConfigParser()
        self.__config.read(conf_file_path)

    def _read(self, _type):
        result = {}
        for option in self.__config.options(_type):
            result[option] = eval(self.__config.get(_type, option))

        for key in result.keys():
            if 'dtype' in key:
                if type(result[key]) == list:
                    result[key] = [DTYPES[dtype] if dtype is not None and dtype in DTYPES.keys() else None
                                   for dtype in result[key]]
                else:
                    result[key] = DTYPES[result[key]] if result[key] is not None and \
                                                         result[key] in DTYPES.keys() else None

            if 'activation' in key:
                if type(result[key]) == list:
                    result[key] = \
                        [ACTIVATIONS[activation_fn] if activation_fn is not None and activation_fn in ACTIVATIONS.keys()
                         else None for activation_fn in result[key]]
                else:
                    result[key] = ACTIVATIONS[result[key]] if result[key] is not None and \
                                                              result[key] in ACTIVATIONS.keys() else None

            if 'initializer' in key:
                parameters = {}
                for temp_key in result.keys():
                    if temp_key.startswith(key) and temp_key != key:
                        parameters[temp_key.replace(key + '_', '')] = result[temp_key]
                if type(result[key]) == list:
                    result[key] = \
                        [INITIALIZERS[initializer]({k: parameters[k][i] for k in parameters.keys()})
                         if initializer is not None and initializer in INITIALIZERS.keys()
                         else None for i, initializer in enumerate(result[key])]
                else:
                    result[key] = INITIALIZERS[result[key]](parameters) if result[key] is not None and \
                                                              result[key] in INITIALIZERS.keys() else None

        return result

    def encoder(self, encoder_type):
        return self._read(encoder_type)

    def loss(self):
        result = {}
        for option in self.__config.options('loss'):
            result[option] = eval(self.__config.get('loss', option))
        if 'loss' in result.keys():
            loss_func = result['loss']
            return LOSSES[loss_func], result
        else:
            raise ValueError

    def optimizer(self):
        result = {}
        for option in self.__config.options('optimizer'):
            result[option] = eval(self.__config.get('optimizer', option))
        if 'optimizer' in result.keys():
            optimizer_func = result['optimizer']
            return OPTIMIZERS[optimizer_func], result
        else:
            raise ValueError

    def accuracy(self):
        result = {}
        for option in self.__config.options('accuracy'):
            result[option] = eval(self.__config.get('accuracy', option))
        if 'accuracy' in result.keys():
            accuracy_func = result['accuracy']
            return ACCURACYS[accuracy_func], result
        else:
            raise ValueError

    def saver(self):
        result = {}
        for option in self.__config.options('saver'):
            result[option] = eval(self.__config.get('saver', option))
        if 'model_name' in result.keys():
            if not os.path.exists(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'models',
                                               result['model_name'])):
                os.mkdir(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'models', result['model_name']))
            result['model_name'] = os.path.join(
                os.path.split(os.path.realpath(__file__))[0], 'models', result['model_name'], result['model_name'])
            return result
        else:
            raise ValueError

    def train(self):
        result = {}
        for option in self.__config.options('train'):
            result[option] = eval(self.__config.get('train', option))
        return result

    def learning_rate(self):
        result = {}
        for option in self.__config.options('learning_rate'):
            result[option] = eval(self.__config.get('learning_rate', option))
        return result
