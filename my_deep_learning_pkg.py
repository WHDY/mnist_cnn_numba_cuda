import numpy as np
from numba import cuda


class FullyConnect(object):
    def __init__(self, inFeatureSzie, outFeatureSize):
        self.weights = np.random.standard_normal((inFeatureSzie, outFeatureSize))/100
        self.biases = np.random.standard_normal(outFeatureSize)/100

        self.g_weights = None
        self.g_biases = None
        self.g_inputs = None

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, gradient_loss_to_this_outputs):
        self.g_inputs = np.dot(gradient_loss_to_this_outputs, np.transpose(self.weights))
        self.g_weights = np.zeros(shape=self.weights.shape, dtype=np.float32)
        self.g_biases = np.zeros(shape=self.biases.shape, dtype=np.float32)
        for i in range(gradient_loss_to_this_outputs.shape[0]):
            self.g_weights += (np.dot(self.inputs[i][:, np.newaxis], gradient_loss_to_this_outputs[i][np.newaxis, :]))
            self.g_biases += gradient_loss_to_this_outputs[i]
        return self.g_inputs

    def update_parameters(self, lr):
        self.weights -= self.g_weights * lr / self.inputs.shape[0]
        self.biases -= self.g_biases * lr / self.inputs.shape[0]


class ReLu(object):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.g_inputs = None

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.outputs = np.maximum(0, inputs)
        return self.outputs

    def backward(self, gradient_loss_to_this_outputs):
        self.g_inputs = gradient_loss_to_this_outputs.copy()
        self.g_inputs[self.inputs < 0] = 0
        return self.g_inputs


class CrossEntropy(object):
    def __init__(self):
        self.softmax = None
        self.labels = None
        self.loss = 0
        self.g_inputs = None

    def forward(self, inputs, labels):
        self.labels = labels
        self.loss = 0
        for i in range(inputs.shape[0]):
            self.loss += (np.sum(np.exp(inputs[i])) - inputs[i, labels[i]])
        self.loss = self.loss / inputs.shape[0]
        self.cal_softmax(inputs)
        return self.loss

    def cal_softmax(self, inputs):
        exp_prediction = np.zeros(inputs.shape, dtype=np.float32)
        self.softmax = np.zeros(inputs.shape, dtype=np.float32)
        for i in range(inputs.shape[0]):
            inputs[i, :] -= np.max(inputs[i, :])
            exp_prediction[i] = np.exp(inputs[i])
            self.softmax[i] = exp_prediction[i] / np.sum(exp_prediction[i])
        return self.softmax

    def backward(self):
        self.g_inputs = self.softmax.copy()
        for i in range(self.g_inputs.shape[0]):
            self.g_inputs[i, self.labels[i]] -= 1
        return self.g_inputs


class MaxPooling2x2(object):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.g_inputs = None

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.outputs = np.zeros(shape=(inputs.shape[0], inputs.shape[1]//2, inputs.shape[2]//2, inputs.shape[3]), dtype=np.float32)
        grid = (inputs.shape[3])
        block = (self.outputs.shape[1], self.outputs.shape[2])
        pool[grid, block](self.inputs, self.outputs)
        return self.outputs

    def backward(self, gradient_loss_this_outputs):
        self.g_inputs = np.zeros(shape=self.inputs.shape, dtype=np.float32)
        grid = (self.outputs.shape[3])
        block = (self.outputs.shape[1], self.outputs.shape[2])
        cal_pool_gradient[grid, block](self.inputs, self.outputs, gradient_loss_this_outputs, self.g_inputs)
        return self.g_inputs


class Conv2D(object):
    def __init__(self, in_channels, kernel_size, features):
        self.features = features
        self.ksize = kernel_size
        weights_scale = np.sqrt(kernel_size * kernel_size * in_channels / 2)
        self.weights = np.random.standard_normal((features, kernel_size, kernel_size, in_channels)) / weights_scale
        self.biases = np.random.standard_normal(features) / weights_scale

        self.g_weights = None
        self.g_biases = None
        self.g_inputs = None
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        self.inputs = np.zeros(shape=(inputs.shape[0], inputs.shape[1]+(self.ksize // 2)*2,
                                      inputs.shape[2] + (self.ksize // 2)*2, inputs.shape[3]), dtype=np.float32)
        self.inputs[:, self.ksize // 2: inputs.shape[1] + self.ksize // 2,
        self.ksize // 2: inputs.shape[2] + self.ksize // 2, :] = inputs.copy()
        self.outputs = np.zeros(shape=(inputs.shape[0], inputs.shape[1], inputs.shape[2], self.features), dtype=np.float32)
        grid = (self.features)
        block = (inputs.shape[1], inputs.shape[2])
        cov[grid, block](self.inputs, self.weights, self.biases, self.outputs)
        return self.outputs

    def backward(self, gradient_loss_to_this_outputs):
        self.g_inputs = np.zeros(shape=self.inputs.shape, dtype=np.float32)
        self.g_weights = np.zeros(self.weights.shape, dtype=np.float32)
        self.g_biases = np.zeros(self.biases.shape, dtype=np.float32)
        grid = (self.features)
        block = (self.inputs.shape[1], self.inputs.shape[2])
        cal_cov_grident[grid, block](self.inputs, self.weights, gradient_loss_to_this_outputs,
                                     self.g_weights, self.g_biases, self.g_inputs)

        self.g_inputs = self.g_inputs[:, self.ksize//2: self.g_inputs.shape[1] - self.ksize//2,
                        self.ksize//2: self.g_inputs.shape[2] - self.ksize//2, :]
        return self.g_inputs

    def update_parameters(self, lr):
        self.weights -= self.g_weights * lr / self.inputs.shape[0]
        self.biases -= self.g_biases * lr / self.inputs.shape[0]


@cuda.jit()
def cov(inputs, weights, biases, outputs):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    f_num = cuda.blockIdx.x

    for n in range(inputs.shape[0]):
        for i in range(weights.shape[1]):
            for j in range(weights.shape[2]):
                for k in range(weights.shape[3]):
                    outputs[n, tx, ty, f_num] += (inputs[n, tx + i, ty + j, k] * weights[f_num, i, j, k])
        outputs[n, tx, ty, f_num] += biases[f_num]


@cuda.jit()
def cal_cov_grident(inputs, weights, gradient_loss_to_this_outputs, g_weights, g_biases, g_inputs):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    f_num = cuda.blockIdx.x

    for n in range(gradient_loss_to_this_outputs.shape[0]):
        for i in range(weights.shape[1]):
            for j in range(weights.shape[2]):
                for k in range(weights.shape[3]):
                    tmp1 = gradient_loss_to_this_outputs[n, tx, ty, f_num] * weights[f_num, i, j, k]
                    tmp2 = gradient_loss_to_this_outputs[n, tx, ty, f_num] * inputs[n, tx+i, ty+j, k]
                    # cuda.atomic.add(g_inputs, (n, tx+i, ty+j, k), tmp1)
                    # cuda.atomic.add(g_weights, (f_num, i, j, k), tmp2)
                    g_inputs[n, tx+i, ty+j, k] += tmp1
                    g_weights[f_num, i, j, k] += tmp2
        # cuda.atomic.add(g_biases, (f_num), gradient_loss_to_this_outputs[n, tx, ty, f_num])
        g_biases[f_num] += gradient_loss_to_this_outputs[n, tx, ty, f_num]


@cuda.jit()
def pool(inputs, outputs):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    d = cuda.blockIdx.x

    for i in range(inputs.shape[0]):
        outputs[i, tx, ty, d] = max(inputs[i, 2 * tx, 2 * ty, d], inputs[i, 2 * tx + 1, 2 * ty, d],
                                    inputs[i, 2 * tx, 2 * ty + 1, d], inputs[i, 2 * tx + 1, 2 * ty + 1, d])


@cuda.jit()
def cal_pool_gradient(inputs, outputs, gradient_to_outputs, grident_to_inputs):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    d = cuda.blockIdx.x

    for k in range(outputs.shape[0]):
        for i in range(2):
            for j in range(2):
                if outputs[k, tx, ty, d] == inputs[k, 2 * tx + i, 2 * ty + j, d]:
                    grident_to_inputs[k, 2 * tx + i, 2 * ty + j, d] = gradient_to_outputs[k, tx, ty, d]
    '''
    for k in range(outputs.shape[0]):
        grident_to_inputs[k, tx*2, ty*2, d] = gradient_to_outputs[k, tx, ty, d]
    '''
