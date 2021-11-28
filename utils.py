import numpy as np
import keras
import sys


def get_type_layers(model):
    """
    get each type of layer name
    :param model:
    :return: dense layer list, convolution layer list
    """
    dense_layer_list = []
    convolution_layer_list = []
    dense_con_layer_list = []
    flatten_layer_list = []
    for layer in model.layers:
        # choose dense, convolution and flatten layer 平坦层
        if isinstance(layer, keras.layers.core.Dense):
            dense_layer_list.append(layer.name)
            dense_con_layer_list.append(layer.name)
        elif isinstance(layer, keras.layers.Conv2D):
            convolution_layer_list.append(layer.name)
            dense_con_layer_list.append(layer.name)
        elif isinstance(layer, keras.layers.core.Flatten):
            flatten_layer_list.append(layer.name)
    return dense_layer_list, convolution_layer_list, dense_con_layer_list, flatten_layer_list


def color_preprocessing(x_train, x_test, mean, std):
    """
    process the input data, scaling, adding bias...
    :param x_train: training data
    :param x_test: testing data
    :param mean: scale
    :param std: bias
    :return: training and testing data after pre-processing
    """
    # astype将输入的多维数据进行指定的类型转换.
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    if len(x_train.shape) == 4:
        for i in range(x_train.shape[3]):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        for i in range(x_test.shape[3]):
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    elif len(x_train.shape) == 3:
        # 将数据集归一化，方便训练，x_train.shape指(10000, 28, 28),x_test.shape指(10000, 28, 28)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
    return x_train, x_test


def model_predict(model, x, y):
    """

    :param model:
    :param x:测试输入
    :param y:测试预言
    :return:准确度
    """
    y_p = model.predict(x)
    # np.argmax返回最大的下一维数据索引
    y_p_class = np.argmax(y_p, axis=1)
    # 把数据降到一维
    correct = np.sum(y.flatten() == y_p_class.flatten())
    acc = float(correct) / len(x)
    return acc


def summary_model(model):
    """
    :param model:
    :return:weight_count, neuron_count, weights_dict(每层对应的权重), neuron_dict(每层对应的神经数)
    """
    weights_dict = {}
    neuron_dict = {}
    weight_count = 0
    neuron_count = 0
    for layer in model.layers:
        # we only calculate dense layer and conv layer 图像数据一般用con2D
        # if isinstance(layer, keras.layers.core.Dense) or isinstance(layer, keras.layers.convolutional._Conv):
        if isinstance(layer, keras.layers.core.Dense) or isinstance(layer, keras.layers.Conv2D):
            w_n = layer.get_weights()[0].size
            n_n = layer.output_shape[-1]
            weight_count += w_n
            neuron_count += n_n
            weights_dict[layer.name] = weight_count
            neuron_dict[layer.name] = neuron_count
    return weight_count, neuron_count, weights_dict, neuron_dict


# 定义一个进度条
def process_bar(num, total):
    rate = float(num) / total
    rate_num = int(100 * rate)
    r = '\r[{}{}]{}%'.format('*' * rate_num, ' ' * (100 - rate_num), rate_num)
    sys.stdout.write(r)
    sys.stdout.flush()
