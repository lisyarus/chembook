import math
import numpy
import numpy.linalg
import sys

_sqrtpi = math.sqrt(math.pi)
_eps = sys.float_info.epsilon


def _choose(n, k):
    r = 1
    for i in range(k):
        r = (r * (n - i)) // (i + 1)
    return r


def _hermite(deg, x):
    h0 = 1.0
    if deg == 0:
        return h0

    h1 = 2.0 * x
    if deg == 1:
        return h1

    for d in range(2, deg+1):
        t = 2.0 * (x * h1 - (d - 1) * h0)
        h0 = h1
        h1 = t

    return h1


def _hermite_coeffs(deg):
    if deg == 0:
        return [1]

    if deg == 1:
        return [0, 2]

    c  = [0] + _hermite_coeffs(deg - 1)
    cc = _hermite_coeffs(deg - 2) + [0, 0]

    return list(2*(a - (deg-1)*b) for (a,b) in zip(c, cc))


# erf(x)/x
# NB: stable for x~0.0
def _erf_over_x(x):
    if abs(x) < _eps:
        return 2.0 / _sqrtpi

    return math.erf(x) / x


# erf(x)/x
# NB: stable for x~0.0
def _erf_over_x(x):
    if abs(x) < _eps:
        return 2.0 / _sqrtpi

    return math.erf(x) / x



def _int_expt2_tn_series(n, a):
    res = 0.0
    s = 1.0
    k = 1
    while True:
        nres = res + 2.0 * s / (2*k - 1 + n)
        if res == nres: break

        res = nres
        s = - s * a / k
        k += 1
    return res
    

def _int_expt2_tn_recursive(n, a):
    if n == 0:
        return math.sqrt(math.pi) * math.erf(math.sqrt(a)) / math.sqrt(a)
    else:
        return -math.exp(-a)/a + (n-1) / (2.0 * a) * _int_expt2_tn_recursive(n-2, a)


# integral of t^n * exp(-a t^2) over [-1, 1]
# n should be >= 0
# NB: the result is 0 if n is odd
# NB: stable for a~0.0
def _int_expt2_tn(n, a):
    if n % 2 == 1:
        return 0.0

    if abs(a) < _eps:
        return  2.0 / (n + 1)
    elif abs(a) < 1.0 / 16.0: # this threshold is somewhat arbitrary
        # use Taylor series
        return _int_expt2_tn_series(n, a)
    else:
        # use recursive formula
        return _int_expt2_tn_recursive(n, a)


def _range3D(n):
    for a in range(n[0]+1):
        for b in range(n[1]+1):
            for c in range(n[2]+1):
                yield (a,b,c)
        

class HGTO1D:

    def __init__(self, sigma, degree, origin):
        self.sigma = sigma
        self.degree = degree
        self.origin = origin

    def __str__(self):
        return "HGTO1D(sigma={},degree={},origin={})".format(self.sigma, self.degree, self.origin)

    def __repr__(self):
        return self.__str__()

    def __call__(self, x):
        tx = (x - self.origin) / self.sigma
        return _hermite(self.degree, tx) * numpy.exp(-tx * tx)


class HGTO:

    def __init__(self, sigma, degree=[0,0,0], origin=[0.0,0.0,0.0]):
        self.sigma = sigma
        self.degree = numpy.array(degree, dtype=int)
        self.origin = numpy.array(origin, dtype=float)

    def __str__(self):
        return "HGTO(sigma={},degree={},origin={})".format(self.sigma, list(self.degree), list(self.origin))

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key):
        return HGTO1D(self.sigma, self.degree[key], self.origin[key])

    def __call__(self, x, y, z):
        return self[0](x) * self[1](y) * self[2](z)


class Nucleus:

    def __init__(self, charge, position=[0.0,0.0,0.0]):
        self.position = numpy.array(position, dtype=float)
        self.charge = charge

    def __str__(self):
        return "Nucleus(charge={}, position={})".format(self.charge, list(self.position))

    def __repr__(self):
        return self.__str__()


def _overlap1d(g1: HGTO1D, g2: HGTO1D) -> float:
    s2 = (g1.sigma ** 2) + (g2.sigma ** 2)
    s = math.sqrt(s2)
    d = g1.origin - g2.origin
    d2 = d ** 2

    return (1.0
            * (g1.sigma ** (g1.degree + 1))
            * (g2.sigma ** (g2.degree + 1))
            / (s ** (g1.degree + g2.degree + 1))
            * _sqrtpi
            * ((-1.0) ** g1.degree)
            * _hermite(g1.degree + g2.degree, d / s)
            * math.exp(- d2 / s2)
        )


# integral of g1·g2
def overlap(g1: HGTO, g2: HGTO) -> float:
    return (1.0
            * _overlap1d(g1[0], g2[0])
            * _overlap1d(g1[1], g2[1])
            * _overlap1d(g1[2], g2[2])
        )


def _kinetic1d(g1: HGTO1D, g2: HGTO1D) -> float:
    return -0.5 * _overlap1d(g1, HGTO1D(g2.sigma, g2.degree + 2, g2.origin)) / (g2.sigma ** 2)


# integral of -0.5*(g1·Δg2)
def kinetic(g1: HGTO, g2: HGTO) -> float:
    o = [
        _overlap1d(g1[0], g2[0]),
        _overlap1d(g1[1], g2[1]),
        _overlap1d(g1[2], g2[2]),
    ]

    k = [
        _kinetic1d(g1[0], g2[0]),
        _kinetic1d(g1[1], g2[1]),
        _kinetic1d(g1[2], g2[2]),
    ]

    return k[0] * o[1] * o[2] + o[0] * k[1] * o[2] + o[0] * o[1] * k[2]


# integral of g1·g2/|r-R|
def nuclear_attraction(g1: HGTO, g2: HGTO, n: Nucleus) -> float:
    s1 = g1.sigma ** 2
    s2 = g2.sigma ** 2

    s = math.sqrt(s1 + s2)

    S = s1 * s2 / (s1 + s2)

    d = g1.origin - g2.origin
    N = math.exp(- numpy.dot(d,d) / (s1 + s2))

    n1 = g1.degree
    n2 = g2.degree

    nsum1 = sum(n1)
    nsum2 = sum(n2)
    nsum  = nsum1 + nsum2

    tc = [0.0] * (2 * nsum + 1)

    D = numpy.array([
        n.position[0] - (g1.origin[0] / s1 + g2.origin[0] / s2) * S,
        n.position[1] - (g1.origin[1] / s1 + g2.origin[1] / s2) * S,
        n.position[2] - (g1.origin[2] / s1 + g2.origin[2] / s2) * S,
    ]) / math.sqrt(S)

    T2 = numpy.dot(D, D)

    for k1 in _range3D(n1):
        for k2 in _range3D(n2):
            ksum1 = sum(k1)
            ksum2 = sum(k2)
            ksum = ksum1 + ksum2
            
            C = 1.0

            C *= _choose(n1[0], k1[0])
            C *= _choose(n1[1], k1[1])
            C *= _choose(n1[2], k1[2])
            C *= _choose(n2[0], k2[0])
            C *= _choose(n2[1], k2[1])
            C *= _choose(n2[2], k2[2])

            if (ksum1 % 2) == 1:
                C = -C

            C /= s ** ksum
            C *= math.sqrt(S) ** (nsum - ksum)
            C /= s1 ** (nsum1 - ksum1)
            C /= s2 ** (nsum2 - ksum2)

            C *= _hermite(k1[0] + k2[0], d[0] / s)
            C *= _hermite(k1[1] + k2[1], d[1] / s)
            C *= _hermite(k1[2] + k2[2], d[2] / s)

            h0 = _hermite_coeffs(n1[0] + n2[0] - k1[0] - k2[0])
            h1 = _hermite_coeffs(n1[1] + n2[1] - k1[1] - k2[1])
            h2 = _hermite_coeffs(n1[2] + n2[2] - k1[2] - k2[2])

            for i0 in range(len(h0)):
                for i1 in range(len(h1)):
                    for i2 in range(len(h2)):
                        p = (nsum - ksum) + i0 + i1 + i2

                        tc[p] += C * h0[i0] * h1[i1] * h2[i2] * (D[0]**i0) * (D[1]**i1) * (D[2]**i2)


    I = 0.0
    for i in range(len(tc)):
        I += tc[i] * _int_expt2_tn(i, T2)

    return - n.charge * N * S * math.pi * (g1.sigma ** nsum1) * (g2.sigma ** nsum2) * I


def nuclear_repulsion(n1: Nucleus, n2: Nucleus) -> float:
    return n1.charge * n2.charge / numpy.linalg.norm(n1.position - n2.position)