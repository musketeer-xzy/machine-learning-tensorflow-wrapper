import tensorflow as tf
from core.base_model import BaseModel


class LSTM(BaseModel):
    @staticmethod
    def build_model(inputs, encoder_config):
        cells = []
        for i in range(encoder_config['layers_num']):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=encoder_config['num_units'][i],
                forget_bias=encoder_config['forget_bias'][i],
                state_is_tuple=encoder_config['state_is_tuple'][i],
                activation=encoder_config['activation'][i],
                reuse=encoder_config['reuse'][i],
                name=encoder_config['name'][i]
            )
            if encoder_config['dropout'][i]:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=lstm_cell,
                    input_keep_prob=encoder_config['input_keep_prob'][i],
                    output_keep_prob=encoder_config['output_keep_prob'][i],
                    state_keep_prob=encoder_config['state_keep_prob'][i],
                    variational_recurrent=encoder_config['variational_recurrent'][i],
                    input_size=encoder_config['input_size'][i],
                    dtype=encoder_config['dtype'][i],
                    seed=encoder_config['seed'][i],
                    dropout_state_filter_visitor=encoder_config['dropout_state_filter_visitor'][i]
                )
            cells.append(lstm_cell)
        multi_lstm_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=encoder_config['multi_state_is_tuple'])
        sequence_length = tf.fill([inputs.get_shape().as_list()[0]], inputs.get_shape().as_list()[1])
        outputs, state = tf.nn.dynamic_rnn(
            cell=multi_lstm_cell,
            inputs=inputs,
            sequence_length=sequence_length,  # encoder_config['sequence_length'],
            initial_state=encoder_config['initial_state'],
            dtype=encoder_config['dynamic_rnn_dtype'],
            parallel_iterations=encoder_config['parallel_iterations'],
            swap_memory=encoder_config['swap_memory'],
            time_major=encoder_config['time_major'],
            scope=encoder_config['scope']
        )
        return outputs

