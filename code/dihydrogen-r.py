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


def make_basis(R, params):
    nuclei = [
        hgto.Nucleus(1.0, [-R/2, 0.0, 0.0]),
        hgto.Nucleus(1.0, [ R/2, 0.0, 0.0]),
    ]

    basis = []

    for sigma in params:
        for n in nuclei:
            basis.append(hgto.HGTO(sigma, origin=n.position))

    return basis, nuclei


def target(R, params):
    energy, coeffs = solve(*equation(*make_basis(R, params)))
    return energy[0]


def optimize(R, params):
    result = scipy.optimize.minimize(lambda params: target(R, params), params, method='Nelder-Mead', options={'adaptive': True})
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


R = numpy.linspace(0.5, 5.0, 101)
E = numpy.zeros(R.shape)

for i in range(len(R)):
    print("{}/{}".format(i, len(R)))
    while True:
        try:
            params = optimize(R[i], numpy.random.uniform(0.5, 5.0, 3))
            break
        except Exception:
            pass

    basis, nuclei = make_basis(R[i], params)
    energy, coeffs = solve(*equation(basis, nuclei))
    
    E[i] = energy[0]

pyplot.plot(R, E)
pyplot.grid()
pyplot.show()
