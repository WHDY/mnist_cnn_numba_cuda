import numpy as np
import time
from read_mnist import DataSet
from my_deep_learning_pkg import FullyConnect, ReLu, CrossEntropy, MaxPooling2x2, Conv2D


def main():
    # read data
    mnistDataSet = DataSet()

    # construct neural network
    conv1 = Conv2D(1, 5, 32)
    reLu1 = ReLu()
    pool1 = MaxPooling2x2()
    conv2 = Conv2D(32, 5, 64)
    reLu2 = ReLu()
    pool2 = MaxPooling2x2()
    fc1 = FullyConnect(7*7*64, 512)
    reLu3 = ReLu()
    fc2 = FullyConnect(512, 10)
    lossfunc = CrossEntropy()

    # train
    lr = 1e-2
    for epoch in range(10):
        for i in range(600):
            train_data, train_label = mnistDataSet.next_batch(100)

            # forward
            A = conv1.forward(train_data)
            A = reLu1.forward(A)
            A = pool1.forward(A)
            A = conv2.forward(A)
            A = reLu2.forward(A)
            A = pool2.forward(A)
            A = A.reshape(A.shape[0], 7*7*64)
            A = fc1.forward(A)
            A = reLu3.forward(A)
            A = fc2.forward(A)
            loss = lossfunc.forward(A, train_label)

            # backward
            grad = lossfunc.backward()
            grad = fc2.backward(grad)
            grad = reLu3.backward(grad)
            grad = fc1.backward(grad)
            grad = grad.reshape(grad.shape[0], 7, 7, 64)
            grad = pool2.backward(grad)
            grad = reLu2.backward(grad)
            grad = conv2.backward(grad)
            grad = grad.copy()
            grad = pool1.backward(grad)
            grad = reLu1.backward(grad)
            grad = conv1.backward(grad)

            # update parameters
            fc2.update_parameters(lr)
            fc1.update_parameters(lr)
            conv2.update_parameters(lr)
            conv1.update_parameters(lr)

            if (i + 1) % 100 == 0:
                test_index = 0
                sum_accu = 0
                for j in range(100):
                    test_data, test_label = mnistDataSet.test_data[test_index: test_index + 100], \
                                            mnistDataSet.test_label[test_index: test_index + 100]
                    A = conv1.forward(test_data)
                    A = reLu1.forward(A)
                    A = pool1.forward(A)
                    A = conv2.forward(A)
                    A = reLu2.forward(A)
                    A = pool2.forward(A)
                    A = A.reshape(A.shape[0], 7 * 7 * 64)
                    A = fc1.forward(A)
                    A = reLu3.forward(A)
                    A = fc2.forward(A)
                    preds = lossfunc.cal_softmax(A)
                    preds = np.argmax(preds, axis=1)
                    sum_accu += np.mean(preds == test_label)
                    test_index += 100
                print("epoch{} train_number{} accuracy: {}%".format(epoch+1, i+1, sum_accu))


if __name__ == "__main__":
    main()
