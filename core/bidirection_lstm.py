import tensorflow as tf
from core.base_model import BaseModel


class BiDirectionLSTM(BaseModel):
    @staticmethod
    def build_model(inputs, encoder_config):
        _inputs = inputs
        outputs = None
        for i in range(encoder_config['layers_num']):
            with tf.variable_scope(None, default_name="bidirectional_rnn"):
                lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(
                    num_units=encoder_config['num_units_fw'][i],
                    forget_bias=encoder_config['forget_bias_fw'][i],
                    state_is_tuple=encoder_config['state_is_tuple_fw'][i],
                    activation=encoder_config['activation_fw'][i],
                    reuse=encoder_config['reuse_fw'][i],
                    name=encoder_config['name_fw'][i]
                )
                if encoder_config['dropout_fw'][i]:
                    lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(
                        cell=lstm_cell_fw,
                        input_keep_prob=encoder_config['input_keep_prob_fw'][i],
                        output_keep_prob=encoder_config['output_keep_prob_fw'][i],
                        state_keep_prob=encoder_config['state_keep_prob_fw'][i],
                        variational_recurrent=encoder_config['variational_recurrent_fw'][i],
                        input_size=encoder_config['input_size_fw'][i],
                        dtype=encoder_config['dtype_fw'][i],
                        seed=encoder_config['seed_fw'][i],
                        dropout_state_filter_visitor=encoder_config['dropout_state_filter_visitor_fw'][i]
                    )
                lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(
                    num_units=encoder_config['num_units_bw'][i],
                    forget_bias=encoder_config['forget_bias_bw'][i],
                    state_is_tuple=encoder_config['state_is_tuple_bw'][i],
                    activation=encoder_config['activation_bw'][i],
                    reuse=encoder_config['reuse_bw'][i],
                    name=encoder_config['name_bw'][i]
                )
                if encoder_config['dropout_bw'][i]:
                    lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(
                        cell=lstm_cell_bw,
                        input_keep_prob=encoder_config['input_keep_prob_bw'][i],
                        output_keep_prob=encoder_config['output_keep_prob_bw'][i],
                        state_keep_prob=encoder_config['state_keep_prob_bw'][i],
                        variational_recurrent=encoder_config['variational_recurrent_bw'][i],
                        input_size=encoder_config['input_size_bw'][i],
                        dtype=encoder_config['dtype_bw'][i],
                        seed=encoder_config['seed_bw'][i],
                        dropout_state_filter_visitor=encoder_config['dropout_state_filter_visitor_bw'][i]
                    )
                sequence_length = tf.fill([_inputs.get_shape().as_list()[0]], _inputs.get_shape().as_list()[1])
                (outputs, state) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_cell_fw,
                    cell_bw=lstm_cell_bw,
                    inputs=_inputs,
                    sequence_length=sequence_length,  # encoder_config['sequence_length'],
                    initial_state_fw=encoder_config['initial_state_fw'],
                    initial_state_bw=encoder_config['initial_state_bw'],
                    dtype=encoder_config['bidirectional_dynamic_rnn_dtype'],
                    parallel_iterations=encoder_config['parallel_iterations'],
                    swap_memory=encoder_config['swap_memory'],
                    time_major=encoder_config['time_major'],
                    scope=encoder_config['scope']
                )
                _inputs = tf.concat(outputs, 2)
        return tf.concat(outputs, 2)

