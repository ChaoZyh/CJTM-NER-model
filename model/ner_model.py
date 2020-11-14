import numpy as np
import os
import tensorflow as tf
import time

from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel
from tensorflow.contrib.layers.python.layers import initializers


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        self.idx_to_tag_boundary = {idx: tag for tag, idx in
                           self.config.vocab_tags_boundary.items()}
        self.initializer = initializers.xavier_initializer()#保持每层梯度相近
        self.num_units = 800
        self.num_head = 8
        self.istrain = True
        self.keep_prob = 0.7


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")
        self.labels_boundary = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels_boundary")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float64, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float64, shape=[],
                        name="lr")


    def get_feed_dict(self, words, labels=None, labels_boundary=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if labels_boundary is not None:
            labels_boundary, _ = pad_sequences(labels_boundary, 0)
            feed[self.labels_boundary] = labels_boundary

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float64,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float64,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                if self.config.char_embeddings is None:
                    self.logger.info("WARNING: randomly initializing char vectors")
                    _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float64,
                        shape=[self.config.nchars, self.config.dim_char])
                else:
                    _char_embeddings = tf.Variable(
                        self.config.char_embeddings,
                        name="_char_embeddings",
                        dtype=tf.float64,
                        trainable=self.config.train_embeddings)

                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float64)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm,
                                              use_peepholes=True,
                                              initializer=self.initializer,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm,
                                              use_peepholes=True,
                                              initializer=self.initializer,
                                              state_is_tuple=True
                                              )
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float64)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = self.self_attention(output)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj_boundary"):
            W_boundary = tf.get_variable("W_boundary", dtype=tf.float64,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags_boundary])

            b_boundary = tf.get_variable("b_boundary", shape=[self.config.ntags_boundary],
                    dtype=tf.float64, initializer=tf.zeros_initializer())

            nsteps_boundary = tf.shape(output)[1]
            output_boundary = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred_boundary = tf.matmul(output_boundary, W_boundary) + b_boundary
            self.logits_boundary = tf.reshape(pred_boundary, [-1, nsteps_boundary, self.config.ntags_boundary])

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float64,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float64, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])


    def self_attention(self, keys, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tf.nn.relu(
                tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            K = tf.nn.relu(
                tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            V = tf.nn.relu(
                tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            Q_ = tf.concat(tf.split(Q, self.num_head, axis=2), axis=0)#划分子张量
            K_ = tf.concat(tf.split(K, self.num_head, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, self.num_head, axis=2), axis=0)
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            key_masks = tf.tile(key_masks, [self.num_head, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)
            query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            query_masks = tf.tile(query_masks, [self.num_head, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            outputs *= query_masks
            if self.istrain:
                outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, self.num_head, axis=0), axis=2)
            outputs += keys
            outputs = self.normalize(outputs)
        return outputs


    def normalize(self, inputs, epsilon=1e-8, scope="ln", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape, dtype=tf.float64), dtype=tf.float64)
            gamma = tf.Variable(tf.ones(params_shape, dtype=tf.float64), dtype=tf.float64)
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta
        return outputs


    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            with tf.variable_scope("ner_crf"):
                log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(
                        tf.cast(self.logits, tf.float32),  self.labels, self.sequence_lengths)
            #self.trans_params = trans_params # need to evaluate it for decoding
            self.ner_loss = tf.reduce_mean(-log_likelihood)
            if self.config.use_muti:
                with tf.variable_scope("boundary_crf"):
                    log_likelihood, self.trans_params_params = tf.contrib.crf.crf_log_likelihood(
                        tf.cast(self.logits_boundary, tf.float32), self.labels_boundary, self.sequence_lengths)
                # self.trans_params_params = trans_params_params # need to evaluate it for decoding
                self.boundary_loss = tf.reduce_mean(-log_likelihood)
                self.loss = 0.8 * self.ner_loss + 0.2 * self.boundary_loss
            else:
                self.loss = self.ner_loss

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]# keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels, labels_boundary) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, labels_boundary, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        """metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]"""

        f1, speed = self.scores(dev)
        return f1, speed


    def scores(self, test):
        addict = {}
        addict["ERR"] = {}
        all_lab = []
        all_label_pred = []
        start_time = time.time()
        for words, labels, labels_boundary in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                all_lab.append(lab)
                all_label_pred.append(lab_pred)
        decode_time = time.time() - start_time
        speed = len(all_lab) / decode_time
        accuracy, precision, recall, f_measure = self.get_ner_fmeasure(all_lab, all_label_pred)
        self.logger.info(
            "all_results  acc:{}, pre:{}, recall:{}, f1:{}".format(accuracy, precision, recall, f_measure))

        return f_measure, speed

    def get_ner_fmeasure(self, golden_lists, predict_lists):
        sent_num = len(golden_lists)
        golden_full = []
        predict_full = []
        right_full = []
        right_tag = 0
        all_tag = 0
        for idx in range(0, sent_num):
            golden_list = golden_lists[idx]
            predict_list = predict_lists[idx]
            for idy in range(len(golden_list)):
                if golden_list[idy] == predict_list[idy]:
                    right_tag += 1
            all_tag += len(golden_list)
            gold_matrix = self.get_ner_BIO(golden_list)
            pred_matrix = self.get_ner_BIO(predict_list)
            right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
            golden_full += gold_matrix
            predict_full += pred_matrix
            right_full += right_ner
        right_num = len(right_full)
        golden_num = len(golden_full)
        predict_num = len(predict_full)
        if predict_num == 0:
            precision = -1
        else:
            precision = (right_num + 0.0) / predict_num
        if golden_num == 0:
            recall = -1
        else:
            recall = (right_num + 0.0) / golden_num
        if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
            f_measure = -1
        else:
            f_measure = 2 * precision * recall / (precision + recall)
        accuracy = (right_tag + 0.0) / all_tag
        print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
        return accuracy, precision, recall, f_measure

    def reverse_style(self, input_string):
        target_position = input_string.index('[')
        input_len = len(input_string)
        output_string = input_string[target_position:input_len] + input_string[0:target_position]
        return output_string

    def get_ner_BIO(self, label_list):
        list_len = len(label_list)
        begin_label = 'B-'
        inside_label = 'I-'
        whole_tag = ''
        index_tag = ''
        tag_list = []
        stand_matrix = []
        for i in range(0, list_len):
            # wordlabel = word_list[i]
            current_label = self.idx_to_tag[label_list[i]].upper()
            if begin_label in current_label:
                if index_tag == '':
                    whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                    index_tag = current_label.replace(begin_label, "", 1)
                else:
                    tag_list.append(whole_tag + ',' + str(i - 1))
                    whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                    index_tag = current_label.replace(begin_label, "", 1)

            elif inside_label in current_label:
                if current_label.replace(inside_label, "", 1) == index_tag:
                    whole_tag = whole_tag
                else:
                    if (whole_tag != '') & (index_tag != ''):
                        tag_list.append(whole_tag + ',' + str(i - 1))
                    whole_tag = ''
                    index_tag = ''
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''

        if (whole_tag != '') & (index_tag != ''):
            tag_list.append(whole_tag)
        tag_list_len = len(tag_list)

        for i in range(0, tag_list_len):
            if len(tag_list[i]) > 0:
                tag_list[i] = tag_list[i] + ']'
                insert_list = self.reverse_style(tag_list[i])
                stand_matrix.append(insert_list)

        return stand_matrix


    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
