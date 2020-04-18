import numpy
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as pyplot
import hgto


def equation(basis):

    B = len(basis)

    H = numpy.zeros((B,B))
    S = numpy.zeros((B,B))

    n = hgto.Nucleus(2.0)

    for i in range(B):
        for j in range(B):
            bi = basis[i]
            bj = basis[j]
            S[i,j] = hgto.overlap(bi, bj)
            H[i,j] = hgto.kinetic(bi, bj) + hgto.nuclear_attraction(bi, bj, n)

    return (H, S)


def solve(H, S):
    return scipy.linalg.eigh(H, S)


def make_basis(params):
    basis = []

    for sigma in params:
        basis.append(hgto.HGTO(sigma))

    return basis


def target(params):
    basis = make_basis(params)
    energy, coeffs = solve(*equation(basis))
    return sum(energy[0:3])


def callback(params):
    print(params)


def optimize(params):
    result = scipy.optimize.minimize(target, params, method='Nelder-Mead', options={'adaptive': True}, callback=callback)
    if not result.success:
        raise RuntimeError("Energy minimization failed")
    return result.x


def zero_like(x):
    if type(x) == numpy.ndarray:
        return numpy.zeros(x.shape)
    elif type(x) == float:
        return 0.0
    else:
        raise RuntimeError("Unknown type {}".format(type(x)))


def wavefunction(basis, coeffs, x, y, z):
    assert len(basis) == len(coeffs)

    f = zero_like(x)
    for i in range(len(basis)):
        f += coeffs[i] * basis[i](x, y, z)
    return f


def density(basis, coeffs, x, y, z):
    f = wavefunction(basis, coeffs, x, y, z)
    return f * f


def radial_density(basis, coeffs, r):
    f = density(basis, coeffs, r, 0, 0)
    return f * numpy.pi * 4 * (r**2)


params = optimize(numpy.random.uniform(0.5, 5.0, 5))

basis = make_basis(params)
energy, coeffs = solve(*equation(basis))
print("Energy:", list(energy))

L = 30
N = 1001
R = numpy.linspace(0, L, N)

for k in range(3):
    state = radial_density(basis, coeffs[:,k], R)
    pyplot.plot(R, state, label="E = {0:.5f}".format(energy[k]))

pyplot.grid()
pyplot.legend()
pyplot.show()