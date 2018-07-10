from config import Config
from core.fullyconnected import FullyConnected
from core.bottleneck import BottleNeck
from core.lstm import LSTM
from core.cnn import CNN
import tensorflow as tf
from datetime import datetime, timedelta
import sys
import os
import numpy as np
import threading
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_logits(inputs, batch_size=-1):
    logits = [inputs]
    #
    # logits.append(FullyConnected.build_model(logits[-1], 'fullyconnected'))
    #
    # logits.append(CNN.build_model(logits[-1], 'cnn'))
    # temp_shape = logits[-1].get_shape()
    # logits.append(tf.reshape(logits[-1], shape=[-1, temp_shape[1] * temp_shape[2] * temp_shape[3]]))
    # logits.append(BottleNeck.build_model(logits[-1], 'bottleneck'))
    # logits.append(FullyConnected.build_model(logits[-1], 'fullyconnected'))

    logits.append(LSTM.build_model(logits[-1], 'lstm'))
    logits.append(tf.reshape(logits[-1], shape=[-1, logits[-1].get_shape()[-1]]))
    # logits.append(BottleNeck.build_model(logits[-1], 'bottleneck'))
    # log(logits[-1].get_shape())
    logits.append(FullyConnected.build_model(logits[-1], 'fullyconnected'))
    logits.append(tf.reshape(logits[-1], shape=[batch_size, -1, 72]))
    logits.append(logits[-1])
    return logits[-1]


def compute_loss(logits, outputs, seq_len=None):
    loss_function, loss_parameter = Config.loss()
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
    accuracy_function, accuracy_parameter = Config.accuracy()
    if accuracy_parameter['accuracy'] == 'ctc':
        accuracy_parameter['seq_len'] = seq_len
        logits = tf.transpose(logits, (1, 0, 2))
    return accuracy_function(labels=outputs, logits=logits, parameters=accuracy_parameter)


def create_global_step():
    return tf.Variable(0, trainable=False, name='global_step')


def create_learning_rate(global_step):
    learning_rate_config = Config.learning_rate()
    learning_rate = tf.train.exponential_decay(learning_rate=learning_rate_config['initial_learning_rate'],
                                               global_step=global_step,
                                               decay_steps=learning_rate_config['decay_steps'],
                                               decay_rate=learning_rate_config['decay_rate'],
                                               staircase=True,
                                               name='learning_rate')
    return learning_rate


def compute_optimizer(loss, learning_rate, global_step):
    optimizer_function, optimizer_parameter = Config.optimizer()
    return optimizer_function(
        loss=loss, learning_rate=learning_rate, global_step=global_step, parameters=optimizer_parameter)


def build_model(batch_size=None):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 150, 20])
    # outputs = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    outputs = tf.sparse_placeholder(dtype=tf.int32)

    seq_len = tf.placeholder(tf.int32, [None])

    logits = compute_logits(inputs, batch_size=batch_size)
    loss = compute_loss(logits, outputs, seq_len=seq_len)
    accuracy = compute_accuracy(logits, outputs, seq_len=seq_len)
    global_step = create_global_step()
    learning_rate = create_learning_rate(global_step)
    optimizer = compute_optimizer(loss, learning_rate, global_step)
    return loss, accuracy, global_step, learning_rate, optimizer, inputs, outputs, seq_len


def build_predict_model(batch_size=1):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 150, 20])
    # outputs = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    # outputs = tf.sparse_placeholder(dtype=tf.int32)

    seq_len = tf.placeholder(tf.int32, [None])

    logits = compute_logits(inputs, batch_size=batch_size)

    predict_value = compute_predict_value(logits, seq_len=seq_len)

    # loss = compute_loss(logits, outputs, seq_len=seq_len)
    # accuracy = compute_accuracy(logits, outputs, seq_len=seq_len)
    # global_step = create_global_step()
    # learning_rate = create_learning_rate(global_step)
    return predict_value, inputs, seq_len


def train(get_next_batch):
    log('开始训练')
    train_config = Config.train()
    _loss, _accuracy, _global_step, _learning_rate, _optimizer, _inputs, _outputs, _seq_len = build_model(batch_size=500)

    saver_config = Config.saver()
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

        batch_id = -1
        avg_loss_list = []
        current_loss = []
        for x in range(1, train_config['max_epoch'] + 1, 1):
            batch_inputs, batch_outputs, batch_id = get_next_batch(batch_id + 1)
            loss, _, global_step, learning_rate = sess.run(
                [_loss, _optimizer, _global_step, _learning_rate],
                feed_dict={_inputs: batch_inputs, _outputs: batch_outputs, _seq_len: np.ones(500) * 150})
            if batch_id == 0 and len(current_loss) > 0:
                avg_loss_list.append((datetime.now(), np.mean(current_loss)))
                t = threading.Thread(target=draw_loss, args=(avg_loss_list,))
                t.start()
                current_loss = []
            current_loss.append(loss)
            log('总迭代 %s\t\tbatch_id %s\t\tloss %.10f\t\t学习率 %.16f' % (global_step, batch_id, loss, learning_rate))
            if x % train_config['save_every_epoch'] == 0:
                saver.save(sess, global_step=_global_step, save_path=model_path)


def predict(inputs):
    _predict, _inputs, _seq_len = build_predict_model(batch_size=len(inputs))

    saver_config = Config.saver()
    saver = tf.train.Saver(max_to_keep=saver_config['max_to_keep'])
    model_path = saver_config['model_name']

    # 训练很多次迭代，每隔10次打印一次loss，可以看情况直接ctrl+c停止
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(os.path.dirname(model_path))
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        predict_value = sess.run([_predict],
                                 feed_dict={_inputs: inputs, _seq_len: np.ones(len(inputs)) * 150})
        return predict_value[0]


def log(message):
    sys.stdout.write('%s\t\t%s\t\t%s' % (datetime.now(), message, os.linesep))
    sys.stdout.flush()


def draw_loss(loss_list):
    plt.figure()
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
    plt.sca(ax)
    plt.cla()
    times, losses = zip(*loss_list)
    x = [(time - times[0]).total_seconds() for time in times]
    y = list(losses)
    time_range = int((times[-1] - times[0]).total_seconds() + 3600 * 48)
    x_ticks = list(range(0, time_range, max(1, time_range // 5)))
    x_labels = [datetime.strftime(times[0] + timedelta(seconds=int(tick)), '%m-%d %H:%M') for tick in x_ticks]
    plt.xticks(x_ticks, x_labels)
    log(x_ticks)
    log(x_labels)
    log(x)
    log(y)
    plt.plot(x, y, label='loss')
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x + [(x[-1] + time_range) / 2, time_range], p(x + [(x[-1] + time_range) / 2, time_range]),
             label='%e x + %e' % (z[0], z[1]))
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    plt.plot(x + [(x[-1] + time_range) / 2, time_range], p(x + [(x[-1] + time_range) / 2, time_range]),
             label='%e x * x + %e' % (z[0], z[1]))
    try:
        popt, pcov = curve_fit(func, x, y)
        plt.plot(x + [(x[-1] + time_range) / 2, time_range], [func(_, popt[0], popt[1], popt[2]) for _ in (
                x + [(x[-1] + time_range) / 2, time_range])],
                 label='%e * e ** (-%e * x) + %e' % (popt[0], popt[1], popt[2]))
    except:
        pass
    plt.grid()
    ax.legend(loc='best')
    plt.savefig('loss.png', format='png')
    plt.close()


def func(x, a, b, c):
    return a * np.exp(-b * x) + c
