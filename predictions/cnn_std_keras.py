# -*- coding: utf-8 -*-
# @Time     :7/17/18 4:01 PM
# @Auther   :Jason Lin
# @File     :cnn_for_cd33$.py
# @Software :PyCharm


# import pickle as pkl
import numpy as np
# import matplotlib.pyplot as plt
# from scipy import interpolate
# from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from keras.models import Model
from keras.models import model_from_yaml
import keras
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import mean_squared_error
# from sklearn import metrics

#np.random.seed(5)
# from tensorflow import set_random_seed
#set_random_seed(12)


def cnn_model(X_train, X_test, y_train, y_test):

    # X_train, y_train = load_data()

    inputs = Input(shape=(1, 23, 4), name='main_input')
    conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(inputs)
    conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(inputs)
    conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(inputs)
    conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(inputs)

    conv_output = keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

    bn_output = BatchNormalization()(conv_output)

    pooling_output = keras.layers.MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

    flatten_output = Flatten()(pooling_output)

    x = Dense(100, activation='relu')(flatten_output)
    x = Dense(23, activation='relu')(x)
    x = keras.layers.Dropout(rate=0.15)(x)

    prediction = Dense(2, name='main_output')(x)

    model = Model(inputs, prediction)

    adam_opt = keras.optimizers.adam(lr = 0.0001)

    model.compile(loss='binary_crossentropy', optimizer = adam_opt)
    print(model.summary())
    model.fit(X_train, y_train, batch_size=100, epochs=200, shuffle=True)

    # later...
    # X_test, y_test = load_crispor_data()
    y_pred = model.predict(X_test).flatten()

def load_model():
    # load YAML and create model
    yaml_file = open('../CNN_std_model/model_cnn_v1.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("../CNN_std_model/model_cnn_v1.h5")
    return loaded_model

def generate_random_target_offtarget(num_mismatches, mutate_PAM_N=False):
    """Generate an array that represents a random guide sequence (23 nt,
    ending in GG) with mismatches in the format used by
    github.com/MichaelLinn/off_target_prediction

    The format is an array of bools with shape 1,1,23,4 and one hot encoding
    for ATGC. Mismatches represented by, e.g. [0,1,1,0] for T/G mismatch.
    Doesn't indicate which is the target seq.

    Doesn't mutate the PAM by default as the N makes little difference, and
    the model hasn't been trained on mutated GGs."""
    # get a random guide sequence
    guide_1d = np.random.randint(0, 3, 21)

    # this obj will record both target and OT
    guide = np.zeros((23, 4), int)
    for i, nti in enumerate(guide_1d):
        guide[i, nti] = 1

    # set the GG
    for i in (-2, -1):
        guide[i, :] = np.array([0, 0, 1, 0])

    # get pos for mm, and add em. Only the first 20 NT can get muted unless mutate_PAM_N is True
    mm_pos = np.random.choice(range(20 + mutate_PAM_N), size=num_mismatches, replace=False)
    for pos in mm_pos:
        available_nt = [i for i, n in enumerate(guide[pos, :]) if not n]
        guide[pos, np.random.choice(available_nt)] = 1
    guide = guide.reshape(1, 1, 23, 4)

    return guide


def get_predictions_for_mismatched_guides(num_guides, max_mm=4):
    model = load_model()

    all_results = {}
    for mm_n in range(1,max_mm+1):
        all_results[mm_n] = res = []
        for _ in range(num_guides):
            input_code = generate_random_target_offtarget(mm_n)
            y_pred = model.predict(input_code).flatten()
            res.append(y_pred[0])
    return all_results

def cnn_predict(guide_seq, off_seq, model):

    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    gRNA_list = list(guide_seq)
    off_list = list(off_seq)
    # print(len(gRNA_list))
    if len(gRNA_list) != len(off_list):
        #raise RuntimeError(f"the length of sgRNA and DNA are not matched! {guide_seq}, {off_seq}")
        return 'Unmatched lengths'

    pair_code = []

    for i in range(len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_list[i]
        gRNA_base_code = code_dict[gRNA_list[i]]
        DNA_based_code = code_dict[off_list[i]]
        pair_code.append(list(np.bitwise_or(gRNA_base_code, DNA_based_code)))
    input_code = np.array(pair_code).reshape(1, 1, 23, 4)
    #print(input_code)

    y_pred = model.predict(input_code).flatten()
    #print(y_pred)
    return y_pred