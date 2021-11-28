import glob
from keras.datasets import mnist
from keras.models import load_model
import keras.backend as K
import gc
import numpy as np
import time
from utils import process_bar


def sort_data(ori_model_path, mutants_path, save_path):
    """
    :param ori_model_path:
    :param mutants_path:
    :param save_path:
    :return:
    """
    (_, __,), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')
    #  图像的 RGB 数值归一,将这些值缩小至 0 到 1 之间，然后将其反馈送到神经网络模型。
    x_test = x_test / 255.
    x_test = x_test.reshape(len(x_test), 28, 28, 1)

    model_path = mutants_path
    model_path = glob.glob(model_path + '/*.h5')
    count_list = [0 for i in range(len(x_test))]
    ori_model = load_model(ori_model_path)
    ori_predict = ori_model.predict(x_test).argmax(axis=-1)
    correct_index = np.where(ori_predict == y_test)[0]
    i = 1
    num = 200
    start_time = time.perf_counter()
    print("开始计算kill_number")
    for path in model_path:
        model = load_model(path)
        result = model.predict(x_test).argmax(axis=-1)
        for index in correct_index:
            if result[index] != ori_predict[index]:
                # 该测试集可以区分变异体
                count_list[index] += 1
        K.clear_session()
        del model
        gc.collect()
        process_bar(i + 1, num)
        i += 1
    elapsed = (time.perf_counter() - start_time)
    print("\nrunning time: ", elapsed)
    count_list = np.asarray(count_list)
    sorted_list = np.argsort(count_list[correct_index])
    # save as npz file
    np.savez(save_path, index=correct_index[sorted_list], kill_num=count_list[correct_index[sorted_list]])
    return


def output_result(result_path):
    data = np.load(result_path)
    kill_num = data["kill_num"]
    res = [0, 0, 0, 0]
    for k_n in kill_num:
        if k_n == 0:
            continue
        elif 0 < k_n <= 50:
            res[0] += 1
        elif k_n <= 100:
            res[1] += 1
        elif k_n <= 150:
            res[2] += 1
        else:
            res[3] += 1
    print("(0,50]:"+str(res[0]))
    print("(50,100]:" + str(res[1]))
    print("(100,150]:" + str(res[2]))
    print("(150,200]:" + str(res[3]))
    return
