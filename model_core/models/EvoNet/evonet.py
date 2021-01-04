# -*- coding: utf-8 -*-

import tensorflow as tf


class EvoNet():
    def __init__(self):
        self.timesteps = None
        self.n_event = None
        self.n_nodes = None
        self.n_features = None
        self.graph_dim = None
        self.initial_node_embedding = None

    def set_configuration(self, model_param, is_training):
        self.timesteps = model_param.his_len
        self.n_event = model_param.n_event
        self.n_nodes = model_param.n_state
        self.n_features = model_param.node_dim
        self.graph_dim = model_param.graph_dim

        self.__GetLearnableParams__(is_training)


    def __weights__(self, input_dim, output_dim, name, init=True, std=0.1, reg=None):
        if init:
            return tf.get_variable(name, shape=[input_dim, output_dim], initializer=tf.random_normal_initializer(0.0, std), regularizer=reg)
        else:
            return tf.get_variable(name, shape=[input_dim, output_dim])

    def __bias__(self, output_dim, name, init=True):
        if init:
            return tf.get_variable(name, shape=[output_dim], initializer=tf.constant_initializer(1.0))
        else:
            return tf.get_variable(name, shape=[output_dim])

    def __GetLearnableParams__(self, is_training):
        # local aggregation - message passing
        self.Win = self.__weights__(self.n_features, self.n_features, name='In_node_weight', init=is_training)
        self.bin = self.__bias__(self.n_features, name='In_node_bias', init=is_training)

        self.Wout = self.__weights__(self.n_features, self.n_features, name='Out_node_weight', init=is_training)
        self.bout = self.__bias__(self.n_features, name='Out_node_bias', init=is_training)

        # attention score
        self.Wa = self.__weights__(self.n_features+self.graph_dim, 1, name='Attention_weight', init=is_training)

        # node lstm
        self.Wi_n = self.__weights__(self.n_features+self.graph_dim, self.n_features, name='Input_node_weight_1', init=is_training)
        self.Ui_n = self.__weights__(self.n_features, self.n_features, name='Input_node_weight_2', init=is_training)
        self.bi_n = self.__bias__(self.n_features, name='Input_node_bias', init=is_training)

        self.Wf_n = self.__weights__(self.n_features+self.graph_dim, self.n_features, name='Forget_node_weight_1', init=is_training)
        self.Uf_n = self.__weights__(self.n_features, self.n_features, name='Forget_node_weight_2', init=is_training)
        self.bf_n = self.__bias__(self.n_features, name='Forget_node_bias', init=is_training)

        self.Wo_n = self.__weights__(self.n_features+self.graph_dim, self.n_features, name='Output_node_weight_1', init=is_training)
        self.Uo_n = self.__weights__(self.n_features, self.n_features, name='Output_node_weight_2', init=is_training)
        self.bo_n = self.__bias__(self.n_features, name='Output_node_bias', init=is_training)

        self.Wc_n = self.__weights__(self.n_features+self.graph_dim, self.n_features, name='Global_node_weight_1', init=is_training)
        self.Uc_n = self.__weights__(self.n_features, self.n_features, name='Global_node_weight_2', init=is_training)
        self.bc_n = self.__bias__(self.n_features, name='Global_node_bias', init=is_training)

        # graph lstm
        self.Wi_g = self.__weights__(self.n_features+self.n_event, self.graph_dim, name='Input_graph_weight_1', init=is_training)
        self.Ui_g = self.__weights__(self.graph_dim, self.graph_dim, name='Input_graph_weight_2', init=is_training)
        self.bi_g = self.__bias__(self.graph_dim, name='Input_graph_bias', init=is_training)

        self.Wf_g = self.__weights__(self.n_features+self.n_event, self.graph_dim, name='Forget_graph_weight_1', init=is_training)
        self.Uf_g = self.__weights__(self.graph_dim, self.graph_dim, name='Forget_graph_weight_2', init=is_training)
        self.bf_g = self.__bias__(self.graph_dim, name='Forget_graph_bias', init=is_training)

        self.Wo_g = self.__weights__(self.n_features+self.n_event, self.graph_dim, name='Output_graph_weight_1', init=is_training)
        self.Uo_g = self.__weights__(self.graph_dim, self.graph_dim, name='Output_graph_weight_2', init=is_training)
        self.bo_g = self.__bias__(self.graph_dim, name='Output_graph_bias', init=is_training)

        self.Wc_g = self.__weights__(self.n_features+self.n_event, self.graph_dim, name='Global_graph_weight_1', init=is_training)
        self.Uc_g = self.__weights__(self.graph_dim, self.graph_dim, name='Global_graph_weight_2', init=is_training)
        self.bc_g = self.__bias__(self.graph_dim, name='Global_graph_bias', init=is_training)

    def MessagePassing(self, send_nodes, receive_nodes, prev_node_embedding):
        """
        Local Structural Information Aggregation
        Implementation: GGNN
        """
        # transition matrix
        Min = tf.matmul(tf.reshape(send_nodes, [-1, self.n_nodes, 1]), tf.reshape(receive_nodes, [-1, 1, self.n_nodes]))
        Mout = tf.matmul(tf.reshape(receive_nodes, [-1, self.n_nodes, 1]), tf.reshape(send_nodes, [-1, 1, self.n_nodes]))

        # middle embedding
        Hin_ = tf.reshape(tf.matmul(Min, prev_node_embedding), [-1, self.n_features])
        Hin = tf.nn.tanh(tf.matmul(Hin_, self.Win) + self.bin)
        Hout_ = tf.reshape(tf.matmul(Mout, prev_node_embedding), [-1, self.n_features])
        Hout = tf.nn.tanh(tf.matmul(Hout_, self.Wout) + self.bout)

        # pooling
        # shape: [-1, n_nodes, n_features]
        H = tf.reduce_max(tf.reshape(tf.concat([Hin, Hout], axis=-1), [-1, self.n_nodes, self.n_features, 2]), axis=-1)

        return H

    def TemporalModeling(self, event, middle_node_emb, prev_graph_emb, prev_graph_mem, prev_node_emb, prev_node_mem):
        """
        Temporal Graph propagation
        :param event: one-hot event marker
        :param middle_node_emb:
        :param prev_node_emb:
        :param prev_graph_emb:
        :return: next embedding
        """

        # attention score
        cur_a = tf.matmul(tf.concat([tf.reduce_mean(middle_node_emb, axis=1), prev_graph_emb], axis=-1), self.Wa)
        # cur_a = tf.ones([tf.shape(middle_node_emb)[0], 1])

        # next node embedding
        g_ = tf.ones([self.batch_size, self.n_nodes, self.graph_dim]) * tf.expand_dims(cur_a * prev_graph_emb, axis=1)
        h_input = tf.concat([g_, middle_node_emb], axis=-1)
        cur_node_emb, cur_node_mem = self.__LSTMUnit__(h_input, prev_node_emb, prev_node_mem, name='node_lstm')

        # next graph embedding
        h_ = cur_a * tf.reduce_mean(cur_node_emb, axis=1)
        g_input = tf.concat([h_, event], axis=-1)
        cur_graph_emb, cur_graph_mem  = self.__LSTMUnit__(g_input, prev_graph_emb, prev_graph_mem, name='graph_lstm')

        return cur_graph_emb, cur_graph_mem, cur_node_emb, cur_node_mem, cur_a

    def __LSTMUnit__(self, input_x, prev_hidden, prev_memory, name):
        """
        LSTM unit
        """
        if name == 'node_lstm':
            input_x = tf.reshape(input_x, [-1, self.n_features+self.graph_dim])
            prev_hidden = tf.reshape(prev_hidden, [-1, self.n_features])
            prev_memory = tf.reshape(prev_memory, [-1, self.n_features])

            # input gate
            I = tf.nn.sigmoid(tf.matmul(input_x, self.Wi_n) + tf.matmul(prev_hidden, self.Ui_n) + self.bf_n)
            # forget gate
            F = tf.nn.sigmoid(tf.matmul(input_x, self.Wf_n) + tf.matmul(prev_hidden, self.Uf_n) + self.bf_n)
            # output gate
            O = tf.nn.sigmoid(tf.matmul(input_x, self.Wo_n) + tf.matmul(prev_hidden, self.Uo_n) + self.bo_n)
            # long term memory cell
            C_ = tf.nn.tanh(tf.matmul(input_x, self.Wc_n) + tf.matmul(F * prev_hidden, self.Uc_n) + self.bc_n)
            # output
            Ct = F * prev_memory + I * C_
            # current information
            current_memory = tf.reshape(Ct, [-1, self.n_nodes, self.n_features])
            current_hidden = tf.reshape(O * tf.nn.tanh(Ct), [-1, self.n_nodes, self.n_features])
        elif name == 'graph_lstm':
            # input gate
            I = tf.nn.sigmoid(tf.matmul(input_x, self.Wi_g) + tf.matmul(prev_hidden, self.Ui_g) + self.bf_g)
            # forget gate
            F = tf.nn.sigmoid(tf.matmul(input_x, self.Wf_g) + tf.matmul(prev_hidden, self.Uf_g) + self.bf_g)
            # output gate
            O = tf.nn.sigmoid(tf.matmul(input_x, self.Wo_g) + tf.matmul(prev_hidden, self.Uo_g) + self.bo_g)
            # long term memory cell
            C_ = tf.nn.tanh(tf.matmul(input_x, self.Wc_g) + tf.matmul(F * prev_hidden, self.Uc_g) + self.bc_g)
            # output
            Ct = F * prev_memory + I * C_
            # current information
            current_memory = Ct
            current_hidden = O * tf.nn.tanh(Ct)
        else:
            raise Exception('no lstm unit.')

        return current_hidden, current_memory

    def Cell(self, send_nodes, receive_nodes, event, prev_graph_emb, prev_graph_mem, prev_node_emb, prev_node_mem):
        with tf.variable_scope('ETNet_Unit'):

            # intermediate node representation
            H_nodes = self.MessagePassing(send_nodes, receive_nodes, prev_node_emb)

            # temporal modeling
            cur_graph_emb, cur_graph_mem, cur_node_emb, cur_node_mem, cur_a = self.TemporalModeling(event, H_nodes, prev_graph_emb, prev_graph_mem, prev_node_emb, prev_node_mem)

        return cur_graph_emb, cur_graph_mem, cur_node_emb, cur_node_mem, cur_a


    def get_embedding(self, state_sequence, event_sequence, initial_node_embedding):
        self.batch_size = tf.shape(state_sequence)[0]
        # time major
        state_sequence = tf.transpose(state_sequence, [1, 0, 2])  # [seq_length * batch_size * n_features]
        event_sequence = tf.transpose(event_sequence, [1, 0, 2])  # [seq_len+1 * batch_size * n_event]

        # inital
        cur_node_emb = tf.ones([self.batch_size, self.n_nodes, self.n_features], dtype=tf.float32) * initial_node_embedding
        cur_node_mem = tf.ones([self.batch_size, self.n_nodes, self.n_features], dtype=tf.float32) * initial_node_embedding
        cur_graph_emb = tf.zeros([self.batch_size, self.graph_dim], tf.float32)
        cur_graph_mem = tf.zeros([self.batch_size, self.graph_dim], tf.float32)

        graph_embs, node_embs, attention_logits = [], [], []

        for i in range(self.timesteps-1):
            cur_graph_emb, cur_graph_mem, cur_node_emb, cur_node_mem, cur_a = self.Cell(state_sequence[i], state_sequence[i+1], event_sequence[i+1], cur_graph_emb, cur_graph_mem, cur_node_emb, cur_node_mem)
            graph_embs.append(cur_graph_emb)
            node_embs.append(cur_node_emb)
            attention_logits.append(cur_a)

        # transpose [batch_size, timestep, *]
        graph_embs = tf.transpose(graph_embs, [1, 0, 2])
        node_embs = tf.transpose(node_embs, [1, 0, 2, 3])
        attention_logits = tf.reshape(tf.transpose(attention_logits, [1, 0, 2]), [-1, self.timesteps-1])

        return graph_embs, node_embs, attention_logits