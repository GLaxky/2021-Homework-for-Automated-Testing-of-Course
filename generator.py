from utils import color_preprocessing, model_predict, summary_model, get_type_layers, process_bar
from keras.datasets import mnist
from keras.models import load_model
from termcolor import colored
import keras.backend as K
import gc
import numpy as np
import random
import time


def cnn_mutants_generation(model, ratio, standard_deviation, operator):
    dense_layer_list, convolution_layer_list, dense_con_layer_list, flatten_layer_list = get_type_layers(model)
    weight_count, neuron_count, weights_dict, neuron_dict = summary_model(model)
    process_weights_num = int(weight_count * ratio) if int(weight_count * ratio) > 0 else 1
    process_neuron_num = int(neuron_count * ratio) if int(neuron_count * ratio) > 0 else 1

    # GF 对所选权重执行基于高斯分布的模糊化
    if operator == 0:
        process_num_list = random_select(weight_count, process_weights_num, dense_con_layer_list, weights_dict)
        for layer_index in range(len(dense_con_layer_list)):
            if process_num_list[layer_index] == 0:
                continue
            layer_name = dense_con_layer_list[layer_index]
            l_weights = model.get_layer(layer_name).get_weights()
            new_l_weights = weights_gaussian_fuzzing(l_weights, process_num_list[layer_index], standard_deviation)
            model.get_layer(layer_name).set_weights(new_l_weights)
    elif operator == 1:
        # WS 随机打乱权重
        process_num_list = random_select(neuron_count, process_neuron_num, dense_con_layer_list, neuron_dict)
        for layer_index in range(len(dense_con_layer_list)):
            if process_num_list[layer_index] == 0:
                continue
            layer_name = dense_con_layer_list[layer_index]
            l_weights = model.get_layer(layer_name).get_weights()
            new_l_weights = weights_shuffle(l_weights, process_num_list[layer_index])
            model.get_layer(layer_name).set_weights(new_l_weights)
    return model


def generator(op, ratio):
    model_path = "model/mnist_lenet5.h5"
    standard_deviation = 0.5
    num = 200
    threshold = 0.9
    save_path = ""
    if op == 0:
        save_path = "save_mutation_GF_" + str(ratio)
    elif op == 1:
        save_path = "save_mutation_WS_" + str(ratio)

    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = color_preprocessing(x_train, x_test, 0, 255)
    x_test = x_test.reshape(len(x_test), 28, 28, 1)

    model = load_model(model_path)
    ori_acc = model_predict(model, x_test, y_test)
    threshold = ori_acc * threshold
    weight_count, neuron_count, weights_dict, neuron_dict = summary_model(model)
    if op == 0:
        print(colored("operator: Gaussian fuzzing (GF)", 'blue'))
    elif op == 1:
        print(colored("operator: Weight Shufﬂe (WS)", 'blue'))
    print(colored("ori acc: %f" % ori_acc, 'blue'))
    print(colored("threshold acc: %f" % threshold, 'blue'))
    print("total weights: ", weight_count)
    print("process weights num: ", int(weight_count * ratio) if int(weight_count * ratio) > 0 else 1)

    # mutants generation
    print("开始生成变异体")
    i = 1
    start_time = time.perf_counter()
    while i <= num:
        if i != 1:
            model = load_model(model_path)
        new_model = cnn_mutants_generation(model, ratio, standard_deviation, op)
        new_acc = model_predict(new_model, x_test, y_test)
        if new_acc < threshold:
            K.clear_session()
            del model
            del new_model
            gc.collect()
            continue
        final_path = ""
        if op == 0:
            final_path = save_path + "/GF" + "_" + str(ratio) + "_" + str(i) + ".h5"
        elif op == 1:
            final_path = save_path + "/WS" + "_" + str(ratio) + "_" + str(i) + ".h5"
        new_model.save(final_path)
        process_bar(i + 1, num)
        i += 1
        K.clear_session()
        del model
        del new_model
        gc.collect()
    elapsed = (time.perf_counter() - start_time)
    print("\nrunning time: ", elapsed)


def random_select(total_num, select_num, layer_list, layer_dict):
    """

    :param total_num:
    :param select_num:
    :param layer_list:
    :param layer_dict:
    :return:
    """
    # numpy.random.choice(a, size=None, replace=True, p=None)
    # 从【0，5）中随机抽取数字，并组成指定大小(size)的数组
    # replace:True表示可以取相同数字，False表示不可以取相同数字
    indices = np.random.choice(total_num, select_num, replace=False)
    process_num_list = []
    process_num_total = 0
    for i in range(len(layer_list)):
        if i == 0:
            num = len(np.where(indices < layer_dict[layer_list[i]])[0])
            process_num_list.append(num)
            process_num_total += num
        else:
            num = len(np.where(indices < layer_dict[layer_list[i]])[0])
            num -= process_num_total
            process_num_total += num
            process_num_list.append(num)
    return process_num_list


def weights_gaussian_fuzzing(weights, process_num, standard_deviation=0.5):
    """

    :param weights:
    :param process_num:
    :param standard_deviation:
    :return:
    """
    weights = weights.copy()
    layer_weights = weights[0]
    weights_shape = layer_weights.shape
    flatten_weights = layer_weights.flatten()
    weights_len = len(flatten_weights)
    weights_select = np.random.choice(weights_len, process_num, replace=False)
    for index in weights_select:
        # np.random.normal()正态分布
        fuzz = np.random.normal(loc=0.0, scale=standard_deviation, size=None)
        flatten_weights[index] = flatten_weights[index] * (1 + fuzz)
    flatten_weights = np.clip(flatten_weights, -1.0, 1.0)
    weights[0] = flatten_weights.reshape(weights_shape)
    return weights


def weights_shuffle(weights, process_num):
    """

    :param weights:
    :param process_num:
    :return:
    """
    weights = weights.copy()
    layer_weights = weights[0].T
    neural_num = len(layer_weights)
    neural_select = random.sample(range(0, neural_num - 1), process_num)
    weights_shape = layer_weights[0].shape
    for neural_index in neural_select:
        flatten_weights = layer_weights[neural_index].flatten()
        # 随机打乱数据方法
        np.random.shuffle(flatten_weights)
        flatten_weights = flatten_weights.reshape(weights_shape)
        layer_weights[neural_index] = flatten_weights
    weights[0] = layer_weights.T
    return weights
