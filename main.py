import numpy as np
import numpy.matlib
from elm import ELM
from numpy import genfromtxt
import random

def split_data(x, t, train_ratio=0.7, test_ratio=0.3):
    assert (type(x) is np.ndarray), "x must be numpy.ndarray type!"
    assert (type(t) is np.ndarray), "t must be numpy.ndarray type!"
    assert (np.shape(x)[0] is np.shape(t)[0]), "Total row of x must be the same as t!"
    assert (np.shape(t) == (np.shape(t)[0],)), "t must have one dimensional column!"
    assert (1 >= train_ratio >= 0), "train_ratio must be between 0 and 1!"
    assert (1 >= test_ratio >= 0), "test_ratio must be between 0 and 1!"
    assert (train_ratio + test_ratio == 1), "The sum of train_ratio and test_ratio must be equal to 1!"
    assert (np.array_equal(np.arange(np.max(np.unique(t))) + 1, np.unique(t).astype(int))), \
        "t elements must be a sequence of number started from 1 until total classes"

    total_features = np.shape(x)[1]

    train_input = np.array([]).reshape(0, total_features)
    train_target = np.array([])
    test_input = np.array([]).reshape(0, total_features)
    test_target = np.array([])

    for i in np.unique(t).astype(int):
        ith_indices = np.where(t == i)[0]
        ith_total_samples = np.shape(ith_indices)[0]
        ith_total_train = int(round(ith_total_samples*train_ratio))
        ith_total_test = ith_total_samples-ith_total_train

        ith_train_input = x[ith_indices[0:ith_total_train], :]
        train_input = np.append(train_input, ith_train_input, axis=0)
        train_target = np.append(train_target, np.zeros(dtype=np.int, shape=ith_total_train) + i, axis=0)

        ith_test_input = x[ith_indices[ith_total_train:], :]
        test_input = np.append(test_input, ith_test_input, axis=0)
        test_target = np.append(test_target, np.zeros(dtype=np.int, shape=ith_total_test) + i, axis=0)
    return train_input, train_target, test_input, test_target


# data preparation
my_data = genfromtxt('iris.csv', delimiter=',')
x_inp = my_data[:, 0:-1]
t_inp = my_data[:, -1]

train_input, train_target, test_input, test_target = split_data(x_inp, t_inp, 0.6, 0.4)

e = ELM(50)
e.train(train_input, train_target)
e.test(test_input, test_target)
print e.train_acc
print e.test_acc

"""
# start for article on https://fennialesmana.com/extreme-learning-machine/
# 1. Prepare the input data (x) and target data (t)
x = np.array([[-1, -5, 5, 5], [2, -4, 2, 3]])
t = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
# 2. Prepare the number of hidden nodes, input weight (w), and bias (b) randomly
w = np.array([[0.5, 0.2], [0.7, -0.4], [-0.6, 0.3]])
b = np.array([[0.6], [0.7], [0.4]])
# 3. Calculate the output of hidden layer (H)
H = np.dot(w, x) + b
H = (1/(1+(numpy.matlib.exp(H*-1)))).transpose()
# 4. Calculate the weight of hidden to output layer using zero error equation
H_inv = np.linalg.pinv(H)
beta = np.dot(H_inv, t.transpose())
#beta2 = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(H), H)), np.transpose(H)), np.transpose(t_inp))
# 5. Calculate the ELM output (o)
output = np.dot(H, beta)
predicted_class = output.argmax(axis=1) + 1
# end for article
"""


