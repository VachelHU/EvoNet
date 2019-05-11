# -*- coding: utf-8 -*-

import numpy as np

class Dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.pointer = 0
        self.num_batch = 0

    # Shuffle the data
    def Shuffle(self, datalength):
        shuffle_indices = np.random.permutation(np.arange(datalength))
        return shuffle_indices

    def SplitBatches(self, data):
        datas = data[:self.num_batch * self.batch_size]
        reminder = data[self.num_batch * self.batch_size:]
        data_batches = np.split(datas, self.num_batch, 0)
        if reminder.shape[0] != 0:
            data_batches.append(reminder)
        return data_batches

    def reset_pointer(self):
        self.pointer = 0

class TraindataLoader(Dataloader):
    def __init__(self, batch_size):
        Dataloader.__init__(self, batch_size)

    def load_data(self, x, y_assign=[], y_reg=[], y_clf=[]):
        self.x = np.array(x, dtype=np.float32)
        self.y_assign_labels = np.array(y_assign, dtype=np.float32)
        self.y_predict_reg_labels = np.array(y_reg, dtype=np.float32)
        self.y_predict_clf_labels = np.array(y_clf, dtype=np.float32)

        # Shuffle the data
        shuffle_indices = self.Shuffle(self.x.shape[0])
        self.x = self.x[shuffle_indices]
        self.y_assign_labels = self.y_assign_labels[shuffle_indices] if self.y_assign_labels.shape[0] != 0 else self.y_assign_labels
        self.y_predict_reg_labels = self.y_predict_reg_labels[shuffle_indices] if self.y_predict_reg_labels.shape[0] != 0 else self.y_predict_reg_labels
        self.y_predict_clf_labels = self.y_predict_clf_labels[shuffle_indices] if self.y_predict_clf_labels.shape[0] != 0 else self.y_predict_clf_labels

        # Split batches
        self.num_batch = int(self.x.shape[0] / self.batch_size)
        self.pointer = 0

        self.x_batches = self.SplitBatches(self.x)
        self.y_assign_batches = self.SplitBatches(self.y_assign_labels) if self.y_assign_labels.shape[0] != 0 else []
        self.y_predict_reg_batches = self.SplitBatches(self.y_predict_reg_labels) if self.y_predict_reg_labels.shape[0] != 0 else []
        self.y_predict_clf_batches = self.SplitBatches(self.y_predict_clf_labels) if self.y_predict_clf_labels.shape[0] != 0 else []
        self.num_batch += 1 if (self.x.shape[0] % self.batch_size) != 0 else 0

    def update_assign_labels(self):
        flag = 1 if (self.x.shape[0] % self.batch_size) != 0 else 0
        self.num_batch -= flag
        self.y_assign_batches = self.SplitBatches(self.y_assign_labels)
        self.num_batch += flag

    def next_batch(self):
        x_batch = self.x_batches[self.pointer]
        y_assign_batch = self.y_assign_batches[self.pointer] if len(self.y_assign_batches) != 0 else []
        y_reg_batch = self.y_predict_reg_batches[self.pointer] if len(self.y_predict_reg_batches) != 0 else []
        y_clf_batch = self.y_predict_clf_batches[self.pointer] if len(self.y_predict_clf_batches) != 0 else []

        self.pointer = (self.pointer + 1) % self.num_batch
        return x_batch, y_assign_batch, y_reg_batch, y_clf_batch

class UseLoader(Dataloader):
    def __init__(self, batch_size):
        Dataloader.__init__(self, batch_size)
        self.flag_y = False

    def load_data(self, x):
        self.instances = np.array(x, dtype=np.float32)

        self.num_batch = int(self.instances.shape[0] / self.batch_size)
        self.pointer = 0

        self.instances_batches = self.SplitBatches(self.instances)
        self.num_batch = len(self.instances_batches)

    def next_batch(self):
        ret = self.instances_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret


class AppLoader(Dataloader):
    def __init__(self, batch_size):
        Dataloader.__init__(self, batch_size)

    def load_data(self, x, y_clf, assign, hidden):
        self.assign = np.array(assign, dtype=np.float32)
        self.hidden = np.array(hidden, dtype=np.float32)
        self.y_clf_labels = np.array(y_clf, dtype=np.float32)
        self.x = np.array(x, dtype=np.float32)

        # Shuffle the data
        shuffle_indices = self.Shuffle(self.assign.shape[0])
        self.assign = self.assign[shuffle_indices]
        # self.hidden = self.hidden[shuffle_indices] if self.hidden.shape[0] != 0 else self.hidden
        self.y_clf_labels = self.y_clf_labels[shuffle_indices] if self.y_clf_labels.shape[0] != 0 else self.y_clf_labels
        self.x = self.x[shuffle_indices]

        # Split batches
        self.num_batch = int(self.assign.shape[0] / self.batch_size)
        self.pointer = 0

        self.assign_batches = self.SplitBatches(self.assign)
        # self.hidden_batches = self.SplitBatches(self.hidden)
        self.x_batches = self.SplitBatches(self.x)
        self.y_clf_batches = self.SplitBatches(self.y_clf_labels) if self.y_clf_labels.shape[0] != 0 else []
        self.num_batch += 1 if (self.x.shape[0] % self.batch_size) != 0 else 0

    def next_batch(self):
        assign_batch = self.assign_batches[self.pointer]
        # hidden_batch = self.hidden_batches[self.pointer]
        hidden_batch = self.hidden
        y_clf_batch = self.y_clf_batches[self.pointer] if len(self.y_clf_batches) != 0 else []
        x_batch = self.x_batches[self.pointer]

        self.pointer = (self.pointer + 1) % self.num_batch
        return x_batch, y_clf_batch, assign_batch, hidden_batch