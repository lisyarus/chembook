import numpy
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as pyplot
import math


def phi(a, x):
    return numpy.exp(-(x-a)**2)


def overlap(a, b):
    return math.sqrt(math.pi / 2) * numpy.exp(- (a - b)**2 / 2)


def potential(a, b):
    return ((a + b)**2 + 1) / 8 * overlap(a,b)


def kinetic(a, b):
    return (1 - (a-b)**2) / 2 * overlap(a,b)


def equation(basis):
    Ai, Aj = numpy.meshgrid(basis, basis)

    H = kinetic(Ai, Aj) + potential(Ai, Aj)
    S = overlap(Ai, Aj)

    return (H, S)


def solve(H, S, eigvals_only=False):
    return scipy.linalg.eigh(H, S, eigvals_only=eigvals_only)


def zero_like(x):
    if type(x) == numpy.ndarray:
        return numpy.zeros(x.shape)
    elif type(x) == float:
        return 0.0
    else:
        raise RuntimeError("Unknown type {}".format(type(x)))


def wavefunction(basis, coeffs, x):
    assert len(basis) == len(coeffs)

    y = zero_like(x)
    for i in range(len(basis)):
        y += coeffs[i] * phi(basis[i], x) 
    return y


def ground_state(x):
    return 1.0 / math.pow(math.pi, 0.25) * numpy.exp(- x**2 / 2.0)


def target(basis):
    energy, coeffs = solve(*equation(basis))
    return energy[0]


def optimize(basis):
    result = scipy.optimize.minimize(target, basis)
    if not result.success:
        raise RuntimeError("Energy minimization failed")
    return result.x


basis = optimize(range(-5, 6))
energy, coeffs = solve(*equation(basis))
print("Energy:", energy)

x = numpy.linspace(-5, 5, 1001)

for k in range(3):
    state = wavefunction(basis, coeffs[:,k], x)
    pyplot.plot(x, state, label="E = {0:.5f}".format(energy[k]))

pyplot.grid()
pyplot.legend()
pyplot.show()