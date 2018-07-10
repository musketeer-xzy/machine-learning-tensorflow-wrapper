from config import Config
from core.fullyconnected import FullyConnected
# from core.bottleneck import BottleNeck
# from core.lstm import LSTM
from core.bidirection_lstm import BiDirectionLSTM
from core.cnn import CNN
import tensorflow as tf
from datetime import datetime, timedelta
import sys
import os
import numpy as np
import threading


config = Config('cnn_bilstm_0613.conf')


def compute_logits(inputs, batch_size=-1):
    #
    # logits.append(FullyConnected.build_model(logits[-1], 'fullyconnected'))
    #
    temp_shape = inputs.get_shape().as_list()
    logits = list([BiDirectionLSTM.build_model(inputs, config.encoder('bidirection_lstm'))])
    logits.append(tf.reshape(logits[-1], shape=[-1, logits[-1].get_shape()[-1]]))
    logits.append(FullyConnected.build_model(logits[-1], config.encoder('fullyconnected_2')))
    logits.append(tf.reshape(logits[-1], shape=[batch_size, -1, 72]))

    return logits[-1], tf.fill([batch_size], temp_shape[1])


def compute_loss(logits, outputs, seq_len=None):
    loss_function, loss_parameter = config.loss()
    if loss_parameter['loss'] == 'ctc':
        loss_parameter['sequence_length'] = seq_len
        logits = tf.transpose(logits, (1, 0, 2))
    return tf.reduce_mean(loss_function(labels=outputs, logits=logits, parameters=loss_parameter))


def compute_predict_value(logits, seq_len=None):
    logits = tf.transpose(logits, (1, 0, 2))
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=True)
    # return tf.sparse_tensor_to_dense(decoded[0], default_value=0)
    return decoded


def compute_accuracy(logits, outputs, seq_len=None):
    accuracy_function, accuracy_parameter = config.accuracy()
    if accuracy_parameter['accuracy'] == 'ctc':
        accuracy_parameter['seq_len'] = seq_len
        logits = tf.transpose(logits, (1, 0, 2))
    return accuracy_function(labels=outputs, logits=logits, parameters=accuracy_parameter)


def create_global_step():
    return tf.Variable(0, trainable=False, name='global_step')


def create_learning_rate():
    learning_rate_config = config.learning_rate()
    learning_rate = tf.Variable(learning_rate_config['initial_learning_rate'], trainable=False, dtype=tf.float32)
    learning_rate_decay_op = learning_rate.assign(learning_rate.value() * learning_rate_config['decay_rate'])
    # learning_rate_enhance_op = learning_rate.assign(learning_rate.value() * 1.1)
    learning_rate_init_op = learning_rate.assign(learning_rate_config['initial_learning_rate'])
    # learning_rate = tf.train.exponential_decay(learning_rate=learning_rate_config['initial_learning_rate'],
    #                                            global_step=global_step,
    #                                            decay_steps=learning_rate_config['decay_steps'],
    #                                            decay_rate=learning_rate_config['decay_rate'],
    #                                            staircase=True,
    #                                            name='learning_rate')
    return learning_rate, learning_rate_decay_op, learning_rate_init_op


def compute_optimizer(loss, learning_rate, global_step):
    optimizer_function, optimizer_parameter = config.optimizer()
    return optimizer_function(
        loss=loss, learning_rate=learning_rate, global_step=global_step, parameters=optimizer_parameter)


def build_model(batch_size=None):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 150, 20])
    # outputs = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    outputs = tf.sparse_placeholder(dtype=tf.int32)

    logits, seq_len = compute_logits(inputs, batch_size=batch_size)

    loss = compute_loss(logits, outputs, seq_len=seq_len)
    accuracy = compute_accuracy(logits, outputs, seq_len=seq_len)
    global_step = create_global_step()
    learning_rate, learning_rate_decay, learning_rate_init = create_learning_rate()
    optimizer = compute_optimizer(loss, learning_rate, global_step)
    return loss, accuracy, global_step, learning_rate, learning_rate_decay, learning_rate_init, optimizer, inputs, outputs


def build_predict_model(batch_size=1):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 150, 20])
    # outputs = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    # outputs = tf.sparse_placeholder(dtype=tf.int32)

    logits, seq_len = compute_logits(inputs, batch_size=batch_size)

    predict_value = compute_predict_value(logits, seq_len=seq_len)

    # loss = compute_loss(logits, outputs, seq_len=seq_len)
    # accuracy = compute_accuracy(logits, outputs, seq_len=seq_len)
    # global_step = create_global_step()
    # learning_rate = create_learning_rate(global_step)
    return predict_value, inputs


def train(get_next_batch, expand_batch):
    batch_size = 250
    log('开始训练')
    train_config = config.train()
    _loss, _accuracy, _global_step, _learning_rate, _learning_rate_decay, _learning_rate_init, _optimizer, _inputs, \
    _outputs = build_model(batch_size=batch_size)

    saver_config = config.saver()
    saver = tf.train.Saver(max_to_keep=saver_config['max_to_keep'])
    model_path = saver_config['model_name']

    # 训练很多次迭代，每隔10次打印一次loss，可以看情况直接ctrl+c停止
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(os.path.dirname(model_path))
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            # step = int(checkpoint.model_checkpoint_path.rsplit('-', 1)[-1])
        else:
            sess.run(tf.global_variables_initializer())
            # step = 0

        mean_loss_list = []
        current_loss = []
        mode = 'w'
        learning_rate = sess.run(_learning_rate_init)
        for x in range(1, train_config['max_epoch'] + 1, 1):
            batch_inputs, batch_outputs, batch_id = get_next_batch()
            if batch_id == 0:
                loss_mean = np.mean(current_loss)
                t = threading.Thread(target=draw_loss, args=(datetime.now(), loss_mean, learning_rate, mode))
                t.start()
                if mode == 'w':
                    mode = 'a'
                mean_loss_list.append(loss_mean)
                if len(mean_loss_list) > 10 and mean_loss_list[-1] > np.mean(mean_loss_list[-10:]):
                    sess.run(_learning_rate_decay)
                if loss_mean < train_config['target_loss']:
                    expand_batch()
                    mode = 'w'
                    # learning_rate = sess.run(_learning_rate_init)
                current_loss = []
            loss, _, global_step, learning_rate = sess.run(
                [_loss, _optimizer, _global_step, _learning_rate],
                feed_dict={_inputs: batch_inputs, _outputs: batch_outputs})
            if loss == float('inf'):
                sess.run(_learning_rate_decay)
            current_loss.append(loss)
            log('总迭代 %s\t\tbatch_id %s\t\tloss %.10f\t\t学习率 %.16f' % (global_step, batch_id, loss, learning_rate))
            if x % train_config['save_every_epoch'] == 0:
                saver.save(sess, global_step=_global_step, save_path=model_path)


def predict(inputs):
    with tf.device("/cpu:0"):
        _predict, _inputs = build_predict_model(batch_size=len(inputs))

        saver_config = config.saver()
        saver = tf.train.Saver(max_to_keep=saver_config['max_to_keep'])
        model_path = saver_config['model_name']

        # 训练很多次迭代，每隔10次打印一次loss，可以看情况直接ctrl+c停止
        with tf.Session() as sess:
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname(model_path))
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
            predict_value = sess.run([_predict], feed_dict={_inputs: inputs})
            return predict_value[0]


def client_predict(batch_size, func_inputs):
    with tf.device("/cpu:0"):
        _predict, _inputs = build_predict_model(batch_size=batch_size)

        saver_config = config.saver()
        saver = tf.train.Saver(max_to_keep=saver_config['max_to_keep'])
        model_path = saver_config['model_name']

        # 训练很多次迭代，每隔10次打印一次loss，可以看情况直接ctrl+c停止
        with tf.Session() as sess:
            checkpoint = tf.train.get_checkpoint_state(os.path.dirname(model_path))
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
            inputs, tasks = func_inputs()
            predict_value = sess.run([_predict], feed_dict={_inputs: inputs})
            return predict_value[0], tasks


def log(message):
    sys.stdout.write('%s\t\t%s\t\t%s' % (datetime.now(), message, os.linesep))
    sys.stdout.flush()


def draw_loss(time, mean_loss, learning_rate, mode):
    with open('loss.txt', mode) as _file:
        _file.write('%s,%s,%s%s' % (time, mean_loss, learning_rate, os.linesep))
