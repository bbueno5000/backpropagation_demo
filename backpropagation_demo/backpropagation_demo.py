"""
TODO: docstring
"""
import numpy

class Backpropagation:
    """
    TODO: docstring
    """
    def __call__(self):
        """
        TODO: docstring
        """
        X = numpy.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
        y = numpy.array([[0], [1], [1], [0]])
        numpy.random.seed(1)
        syn0 = 2 * numpy.random.random((3, 4)) - 1
        syn1 = 2 * numpy.random.random((4, 1)) - 1
        for j in range(60000):
            k0 = X
            k1 = self.nonlin(numpy.dot(k0, syn0))
            k2 = self.nonlin(numpy.dot(k1, syn1))
            k2_error = y - k2
            if j % 10000 == 0:
                print('Error:' + str(numpy.mean(numpy.abs(k2_error))))
            k2_delta = k2_error * self.nonlin(k2, deriv=True)
            k1_error = k2_delta.dot(syn1.T)
            k1_delta = k1_error * self.nonlin(k1, deriv=True)
            syn1 += k1.T.dot(k2_delta)
            syn0 += k0.T.dot(k1_delta)
    
    def nonlin(self, x, deriv=False):
        """
        TODO: docstring
        """
        if deriv:
            return x * (1 - x)
        return 1 / (1 + numpy.exp(-x))

if __name__ == '__main__':
    backpropagation = Backpropagation()
    backpropagation()
