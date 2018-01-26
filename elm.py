import numpy as np
import numpy.matlib
import random


class ELM:
    def __init__(self, max_iteration=100, total_hidden_nodes=-1, weight_lower_bound=-1, weight_upper_bound=1):
        self.max_iteration = max_iteration
        self.total_train_samples = 0
        self.total_test_samples = 0
        self.total_features = 0
        self.total_classes = 0
        self.total_hidden_nodes = total_hidden_nodes
        self.weight_lower_bound = weight_lower_bound
        self.weight_upper_bound = weight_upper_bound
        self.w = np.array([])
        self.bias = np.array([])
        self.beta = np.array([])

        self.train_acc = 0
        self.test_acc = 0
        self.is_trained = False

    def train(self, x=np.array([]), t=np.array([])):
        assert (type(x) is np.ndarray), "x must be numpy.ndarray type!"
        assert (type(t) is np.ndarray), "t must be numpy.ndarray type!"
        assert (np.shape(x)[0] is np.shape(t)[0]), "Total row of x must be the same as t!"
        assert (np.shape(t) == (np.shape(t)[0], )), "t must have one dimensional column!"

        # variable initialization
        self.total_classes = np.max(np.unique(t))
        assert (np.array_equal(np.arange(self.total_classes)+1, np.unique(t).astype(int))), \
            "t elements must be a sequence of number started from 1 until total classes"

        x_train = x.transpose()
        t_train = (np.arange(self.total_classes) == t[:, None]-1).astype(int)

        self.total_features = np.shape(x_train)[0]
        self.total_train_samples = np.shape(x_train)[1]

        if self.total_hidden_nodes == -1:
            self.total_hidden_nodes = random.randint(1, self.total_train_samples)

        self.w = np.random.uniform(low=self.weight_lower_bound, high=self.weight_upper_bound,
                                   size=(self.total_hidden_nodes, self.total_features))
        self.bias = np.random.uniform(low=-self.weight_lower_bound, high=self.weight_upper_bound,
                                      size=(self.total_hidden_nodes, 1))

        # calculate matrix H, apply activation function, and calculate b
        H = np.dot(self.w, x_train) + self.bias
        H = (1/(1+(numpy.matlib.exp(H*-1)))).transpose()

        self.beta = np.dot(np.linalg.pinv(H), t_train)

        # training accuracy calculation
        output = np.dot(H, self.beta)  # total output class * total hidden node
        predicted_class = output.argmax(axis=1) + 1
        self.train_acc = float(np.sum((t == predicted_class).astype(int)))/float(self.total_train_samples)*100
        self.is_trained = True

    def test(self, x=np.array([]), t=np.array([])):
        assert (self.is_trained is True), "ELM must have been trained first!"
        assert (type(x) is np.ndarray), "x must be numpy.ndarray type!"
        assert (type(t) is np.ndarray), "t must be numpy.ndarray type!"
        assert (np.shape(x)[0] is np.shape(t)[0]), "Total row of x must be the same as t!"
        assert (np.shape(t) == (np.shape(t)[0],)), "t must have one dimensional column!"
        assert (np.array_equal(np.arange(self.total_classes)+1, np.unique(t).astype(int))), \
            "t elements must be a sequence of number started from 1 until total classes"

        # variable initialization
        x_test = x.transpose()
        self.total_test_samples = np.shape(x_test)[1]

        # calculate matrix H and apply activation function
        H = np.dot(self.w, x_test) + self.bias
        H = (1/(1+(numpy.matlib.exp(H*-1)))).transpose()

        # test accuracy calculation
        output = np.dot(H, self.beta)
        predicted_class = output.argmax(axis=1) + 1
        self.test_acc = float(np.sum((t == predicted_class).astype(int))) / float(self.total_test_samples) * 100














