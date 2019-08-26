import random
import numpy as np
import matplotlib.pyplot as plt


def acfun(inputs):
    return 1.0 / (1.0 + np.e ** (-inputs))


def d_acfun(inputs):
    return acfun(inputs) * (1.0 - acfun(inputs))


def cost(outputs, label):
    return (outputs - label) ** 2


def d_cost(outputs, label):
    return 2 * (outputs - label)


class unit(object):

    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.w1 = np.random.normal(0, 1, 1)[0]
        self.w2 = np.random.normal(0, 1, 1)[0]

    def cal(self, inputs):
        self.inputs = inputs
        self.outputs = acfun(inputs * self.w1) * self.w2 + inputs
        return self.outputs


def res(depth):
    res_list = []
    for i in range(depth):
        res_list.append(unit())
    return res_list


def forward(res_list, inputs):
    for ele in res_list:
        inputs = ele.cal(inputs)
    return inputs


def d_layer(unit):
    w1, w2 = unit.w1, unit.w2
    inputs = unit.inputs
    return 1.0 + w1 * w2 * d_acfun(w1 * inputs)


def d_w1(unit):
    w1, w2 = unit.w1, unit.w2
    inputs = unit.inputs
    return inputs * w2 * d_acfun(w1 * inputs)


def d_w2(unit):
    return acfun(unit.w1 * unit.inputs)


def updata_w1(gradient_cost, res_list, index):
    if index == -1:
        return gradient_cost * d_w1(res_list[index])
    else:
        middle_gradient = 1
        for i in range(-1, index, -1):
            middle_gradient *= d_layer(res_list[i])
        return gradient_cost * d_w1(res_list[index]) * middle_gradient


def updata_w2(gradient_cost, res_list, index):
    if index == -1:
        return gradient_cost * d_w2(res_list[index])
    else:
        middle_gradient = 1
        for i in range(-1, index, -1):
            middle_gradient *= d_layer(res_list[i])
        return gradient_cost * d_w2(res_list[index]) * middle_gradient


def backward(res_list, label, eta):
    gradient_cost = d_cost(res_list[-1].outputs, label)
    for index in range(-1, -len(res_list)-1, -1):
        res_list[index].w1 -= eta * updata_w1(gradient_cost, res_list, index)
        res_list[index].w2 -= eta * updata_w2(gradient_cost, res_list, index)


def generate_data(a, b):
    data, label = [], []
    for i in range(int(2e4)):
        ele = random.uniform(a, b)
        data.append(ele)
        label.append(0.5 * ele ** 2 + 0.5)
    return data, label


def train(lr, epochs, depth):
    error = []
    test = [random.uniform(0, 1) for i in range(2000)]
    lr, epochs = lr, epochs
    res_list = res(depth)
    data, label = generate_data(0, 1)
    for epoch in range(int(epochs)):
        print('epoch : ', epoch, end='')
        for index in range(len(data)):
            forward(res_list, data[index])
            backward(res_list, label[index], lr)
            error_add = 0
            result_add = 0
        for ele in test:
            result = forward(res_list, ele)
            result_add += result
            # print('input : %.5f' % ele, 'label : %.5f' % (0.5 * ele ** 2 + 0.5), '  predict : %.5f' % result,
            #       '  error : %.5f' % (abs(result - 0.5 * ele ** 2 - 0.5)))
            error_add += abs(result - 0.5 * ele ** 2 - 0.5)
        print('   residual rate : %.4f' % (error_add / result_add * 100), '%')
        error.append(error_add / 2000)
    print('\n')
    for index, ele in enumerate(res_list):
        print('layer ' + str(index) + ' : ', 'activation function : sigmoid', '  w1 : %3.5f' % ele.w1, '  w2 : %3.5f' % ele.w2)

    plt.title('Error -- Epoch')
    plt.xlabel('epoch')
    plt.ylabel('mean error')
    plt.plot(range(len(error)), error)
    plt.show()


if __name__ == '__main__':
    train(lr=1e-2, epochs=100, depth=5)
