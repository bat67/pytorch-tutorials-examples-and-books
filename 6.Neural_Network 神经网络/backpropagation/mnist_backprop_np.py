import  random
import  numpy as np
import  mnist



def sigmoid(z):
    """
    The sigmoid function.
     [30/10, 1]
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


class Network:

    def __init__(self, sizes):
        """
        The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.
        :param sizes: [784, 100, 10]
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # [ch_out, 1]
        self.biases = [np.random.randn(ch_out, 1) for ch_out in sizes[1:]]
        # [ch_out, ch_in]
        self.weights = [np.random.randn(ch_out, ch_in)
                            for ch_in, ch_out in zip(sizes[:-1], sizes[1:])]

    def forward(self, x):
        """

        :param x: [784, 1]
        :return: [30, 1]
        """

        for b, w in zip(self.biases, self.weights):
            # [30, 784]@[784, 1] + [30, 1]=> [30, 1]
            # [10, 30]@[30, 1] + [10, 1]=> [10, 1]
            x = sigmoid(np.dot(w, x)+b)
        return x

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            # for every (x,y)
            for mini_batch in mini_batches:
                loss = self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}, Loss: {3}".format(
                    j, self.evaluate(test_data), n_test, loss))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate.
        """
        # https://en.wikipedia.org/wiki/Del
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        loss = 0

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w, loss_ = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            loss += loss_

        # tmp1 = [np.linalg.norm(b/len(mini_batch)) for b in nabla_b]
        # tmp2 = [np.linalg.norm(w/len(mini_batch)) for w in nabla_w]
        # print(tmp1)
        # print(tmp2)

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        loss = loss / len(mini_batch)

        return loss

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 1. forward
        activation = x
        # w*x = z => sigmoid => x/activation
        zs = [] # list to store all the z vectors, layer by layer
        activations = [x] # list to store all the activations, layer by layer
        for b, w in zip(self.biases, self.weights):
            # https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication
            # np.dot vs np.matmul = @ vs element-wise *
            z = np.dot(w, activation)
            z = z + b # [256, 784] matmul [784] => [256]
            # [256] => [256, 1]
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        loss = np.power(activations[-1]-y, 2).sum()

        # 2. backward
        # (Ok-tk)*(1-Ok)*Ok
        # [10] - [10] * [10]
        delta = self.cost_prime(activations[-1], y) * sigmoid_prime(zs[-1]) # sigmoid(z)*(1-sigmoid(z))
        # O_j*Delta_k
        # [10]
        nabla_b[-1] = delta
        # deltaj * Oi
        # [10] @ [30, 1]^T => [10, 30]
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            # [30, 1]
            z = zs[-l]
            sp = sigmoid_prime(z)
            # sum()
            # [10, 30] => [30, 10] @ [10, 1] => [30, 1] * [30, 1]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return nabla_b, nabla_w, loss

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_prime(self, output_activations, y):
        """
        Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        """
        return output_activations-y




def main():

    x_train, y_train, x_test, y_test = mnist.load_data(reshape=[784,1])
    # (50000, 784) (50000, 10) (10000, 784) (10000, 10)
    print('x_train, y_train, x_test, y_test:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    np.random.seed(66)

    model = Network([784, 30, 10])
    data_train = list(zip(x_train, y_train))
    data_test = list(zip(x_test, y_test))
    model.SGD(data_train, 10000, 10, 0.1, data_test)



if __name__ == '__main__':
    main()