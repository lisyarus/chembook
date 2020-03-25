import numpy
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as pyplot
import hgto


def equation(basis, nuclei):

    B = len(basis)

    H = numpy.zeros((B,B))
    S = numpy.zeros((B,B))

    total_nuclear_repulsion = 0.0
    for i in range(len(nuclei)):
        for j in range(i+1, len(nuclei)):
            total_nuclear_repulsion += hgto.nuclear_repulsion(nuclei[i], nuclei[j])

    for i in range(B):
        for j in range(B):
            bi = basis[i]
            bj = basis[j]
            S[i,j] = hgto.overlap(bi, bj)
            H[i,j] = hgto.kinetic(bi, bj)
            for n in nuclei:
                H[i,j] += hgto.nuclear_attraction(bi, bj, n)
            H[i,j] += S[i,j] * total_nuclear_repulsion

    return (H, S)


def solve(H, S):
    return scipy.linalg.eigh(H, S)


def make_basis(params):
    R = 1.459

    nuclei = [
        hgto.Nucleus(2.0, [-R/2, 0.0, 0.0]),
        hgto.Nucleus(1.0, [ R/2, 0.0, 0.0]),
    ]

    N = len(params) // 2

    sigma_He = params[0:N]
    sigma_H = params[N:]

    basis = []

    for sigma in sigma_He:
        basis.append(hgto.HGTO(sigma, origin=nuclei[0].position))

    for sigma in sigma_H:
        basis.append(hgto.HGTO(sigma, origin=nuclei[1].position))

    return basis, nuclei


def target(params):
    energy, coeffs = solve(*equation(*make_basis(params)))
    return energy[0]


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
    assert len(basis) == len(coeffs)

    f = zero_like(x)
    for i in range(len(basis)):
        for j in range(len(basis)):
            f += coeffs[i] * coeffs[j] * basis[i](x, y, z) * basis[j](x, y, z)
    return f


params = optimize(numpy.random.uniform(0.5, 5.0, 4))

basis, nuclei = make_basis(params)
energy, coeffs = solve(*equation(basis, nuclei))
print("Energy:", list(energy))

L = 4
N = 1001
X = numpy.linspace(-L, L, N)

for k in [0]:
    state = wavefunction(basis, coeffs[:,k], X, 0, 0)
    pyplot.plot(X, state, label="E = {0:.5f}".format(energy[k]))

pyplot.grid()
pyplot.legend()
pyplot.show()