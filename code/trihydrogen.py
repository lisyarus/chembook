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
    A = 1.7
    R = A / numpy.sqrt(3.0)

    nuclei = [
        hgto.Nucleus(1.0, [   R,  0.0, 0.0]),
        hgto.Nucleus(1.0, [-R/2,  A/2, 0.0]),
        hgto.Nucleus(1.0, [-R/2, -A/2, 0.0]),
    ]

    basis = []

    for sigma in params:
        for n in nuclei:
            basis.append(hgto.HGTO(sigma, origin=n.position))

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
    f = wavefunction(basis, coeffs, x, y, z)
    return f * f


params = optimize(numpy.random.uniform(0.5, 5.0, 3))

basis, nuclei = make_basis(params)
energy, coeffs = solve(*equation(basis, nuclei))
print("Energy:", list(energy))

for k in [1,2]:
    c = coeffs[:,k]
    for i in range(len(params)):
        print(sum(c[3*i:3*i+3]))

L = 3
N = 101
X = numpy.linspace(-L, L, N)
Y = numpy.linspace(-L, L, N)

X, Y = numpy.meshgrid(X, Y)

figure = pyplot.figure()
for k in range(3):
    state = wavefunction(basis, coeffs[:,k], X, Y, 0)
    if k == 0:
        state = -state
    figure.add_subplot(1, 3, k+1)
    pyplot.imshow(state, vmin=-0.4, vmax=0.4)
    pyplot.grid()

pyplot.show()