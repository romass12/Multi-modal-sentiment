import pickle
import sys

import keras
import numpy as np


def createOneHotMosei3way(train_label, test_label):
    train = np.zeros((train_label.shape[0], train_label.shape[1],3))
    test = np.zeros((test_label.shape[0], test_label.shape[1],3))

    return train, test


def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def get_raw_data(data, classes):
    mode = 'audio'
    with open('./dataset/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        audio_train, train_label, _, _, audio_test, test_label, _, train_length, _, test_length, _, _,_ = u.load()
        print("Shape for audio test data {}".format(audio_test.shape))

    mode = 'text'
    with open('./dataset/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        text_train, train_label, _, _, text_test, test_label, _, train_length, _, test_length, _, _, _ = u.load()
        print("Shape for text test data {}".format(text_test.shape))

    mode = 'video'
    with open('./dataset/{1}_{2}way.pickle'.format(data, mode, classes), 'rb') as handle:
        u = pickle._Unpickler(handle)
        u.encoding = 'latin1'
        video_train, train_label, _, _, video_test, test_label, _, train_length, _, test_length, _, _,_ = u.load()
        print("Shape for video test data {}".format(video_test.shape))

    print('audio train shape is {}'.format(audio_train.shape))
    print('audio test shape is {}'.format(audio_test.shape))

    train_data = np.concatenate((audio_train, video_train, text_train), axis=-1)
    test_data = np.concatenate((audio_test, video_test, text_test), axis=-1)

    train_label = train_label.astype('int')
    test_label = test_label.astype('int')
    print("Trimodal Train data shape {}".format(train_data.shape))
    print("Trimodal Test data shape {}".format(test_data.shape))
    train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
    
    for i in range(len(train_length)):
        train_mask[i, :train_length[i]] = 1.0

    test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
    for i in range(len(test_length)):
        test_mask[i, :test_length[i]] = 1.0

    train_label, test_label = createOneHotMosei3way(train_label, test_label)
    seqlen_train = train_length
    seqlen_test = test_length

    return train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask