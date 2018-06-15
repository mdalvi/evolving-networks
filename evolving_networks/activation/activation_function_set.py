from evolving_networks.activation.abs import Abs
from evolving_networks.activation.arc_tan import ArcTan
from evolving_networks.activation.bent import Bent
from evolving_networks.activation.binary_step import BinaryStep
from evolving_networks.activation.clamped import Clamped
from evolving_networks.activation.elu import Elu
from evolving_networks.activation.exp import Exp
from evolving_networks.activation.gaussian import Gaussian
from evolving_networks.activation.hat import Hat
from evolving_networks.activation.identity import Identity
from evolving_networks.activation.inverse import Inverse
from evolving_networks.activation.inverse_square_root_linear_unit import InverseSquareRootLinearUnit
from evolving_networks.activation.inverse_square_root_unit import InverseSquareRootUnit
from evolving_networks.activation.leaky_relu import LeakyRelu
from evolving_networks.activation.relu import Relu
from evolving_networks.activation.selu import Selu
from evolving_networks.activation.sigmoid import Sigmoid
from evolving_networks.activation.sin import Sin
from evolving_networks.activation.sinc import Sinc
from evolving_networks.activation.soft_plus import SoftPlus
from evolving_networks.activation.soft_sign import SoftSign
from evolving_networks.activation.tanh import Tanh
from evolving_networks.errors import InvalidActivationError


class ActivationFunctionSet(object):
    def __init__(self):
        self.functions = {}
        self._add('sigmoid', Sigmoid)
        self._add('tanh', Tanh)
        self._add('sin', Sin)
        self._add('gauss', Gaussian)
        self._add('relu', Relu)
        self._add('softplus', SoftPlus)
        self._add('identity', Identity)
        self._add('clamped', Clamped)
        self._add('inv', Inverse)
        self._add('exp', Exp)
        self._add('abs', Abs)
        self._add('hat', Hat)
        self._add('step', BinaryStep)
        self._add('arctan', ArcTan)
        self._add('softsign', SoftSign)
        self._add('isrua', InverseSquareRootUnit)
        self._add('lrelu', LeakyRelu)
        self._add('elu', Elu)
        self._add('selu', Selu)
        self._add('isrlua', InverseSquareRootLinearUnit)
        self._add('bent', Bent)
        self._add('sinc', Sinc)

    def _add(self, name, func):
        self.functions[name] = func

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationError("NO SUCH ACTIVATION FUNCTION FOUND [{}]".format(name))
        return f

    def is_valid(self, name):
        return name in self.functions
