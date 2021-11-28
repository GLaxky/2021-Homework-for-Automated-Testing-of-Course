from generator import generator
from sort_data import sort_data, output_result

if __name__ == '__main__':
    model_path = "model/mnist_lenet5.h5"
    save_path = "result_of_kill_number/"

    generator(0, 0.05)
    sort_data(model_path, "save_mutation_GF_0.05", save_path+"lenet5-gf-0.05.npz")
    print("ratio=0.05, GF")
    output_result(save_path+"lenet5-gf-0.05.npz")

    # print()
    # generator(0, 0.03)
    # sort_data(model_path, "save_mutation_GF_0.03", save_path + "lenet5-gf-0.03.npz")
    # print("ratio=0.03, GF")
    # output_result(save_path + "lenet5-gf-0.03.npz")
    #
    # print()
    # generator(0, 0.01)
    # sort_data(model_path, "save_mutation_GF_0.01", save_path + "lenet5-gf-0.01.npz")
    # print("ratio=0.01, GF")
    # output_result(save_path + "lenet5-gf-0.01.npz")
    #
    # print()
    # generator(1, 0.05)
    # sort_data(model_path, "save_mutation_WS_0.05", save_path + "lenet5-ws-0.05.npz")
    # print("ratio=0.05, WS")
    # output_result(save_path + "lenet5-ws-0.05.npz")
    #
    # print()
    # generator(1, 0.03)
    # sort_data(model_path, "save_mutation_WS_0.03", save_path + "lenet5-ws-0.03.npz")
    # print("ratio=0.03, WS")
    # output_result(save_path + "lenet5-ws-0.03.npz")
    #
    # print()
    # generator(1, 0.01)
    # sort_data(model_path, "save_mutation_WS_0.01", save_path + "lenet5-ws-0.01.npz")
    # print("ratio=0.01, WS")
    # output_result(save_path + "lenet5-ws-0.01.npz")

