import numpy as np
from scipy.interpolate import CubicSpline

z, b0tab, b1tab, db0tab, db1tab = np.loadtxt("./b0b1.dat", usecols=(0, 1, 2, 3, 4)).T
b0 = CubicSpline(z, b0tab)
b1 = CubicSpline(z, b1tab)
db0 = CubicSpline(z, db0tab)
db1 = CubicSpline(z, db1tab)


def kep(psi):
    return psi - 0.0167 * np.sin(psi)


_psi = np.linspace(0, 2 * np.pi, 1000)
_phi = kep(_psi)
getpsi = lambda x: np.interp(x, _phi, _psi)


def plx_config(t0, alpha, delta):
    t0par = t0
    ecc = 0.0167
    vernal = 2719.55
    offset = 75
    radian = 180 / np.pi
    p = 2 * np.pi
    peri = vernal - offset  # Perihelion time icrs frame

    spring = np.array([1.0, 0.0, 0.0])
    summer = np.array([0.0, 0.9174, 0.3971])
    north = np.array([0, 0, 1])
    star = np.array(
        [
            np.cos(alpha / radian) * np.cos(delta / radian),
            np.sin(alpha / radian) * np.cos(delta / radian),
            np.sin(delta / radian),
        ]
    )
    east = np.cross(north, star)
    east = east / np.sum(east**2) ** 0.5
    north = np.cross(star, east)

    phi = (1 - offset / 365.25) * p % p
    psi = getpsi(phi)
    cos = (np.cos(psi) - ecc) / (1 - ecc * np.cos(psi))
    sin = -((1 - cos**2) ** 0.5)
    xpos = spring * cos + summer * sin
    ypos = -spring * sin + summer * cos

    phi = (t0 + 1 - peri) / 365.25 * p % p
    psi = getpsi(phi)
    sun = xpos * (np.cos(psi) - ecc) + ypos * (np.sin(psi)) * (1 - ecc**2) ** 0.5
    qn2 = np.dot(sun, north)
    qe2 = np.dot(sun, east)

    phi = (t0 - 1 - peri) / 365.25 * p % p
    psi = getpsi(phi)
    sun = xpos * (np.cos(psi) - ecc) + ypos * (np.sin(psi)) * (1 - ecc**2) ** 0.5
    qn1 = np.dot(sun, north)
    qe1 = np.dot(sun, east)

    phi = (t0 - peri) / 365.25 * p % p
    psi = getpsi(phi)
    sun = xpos * (np.cos(psi) - ecc) + ypos * (np.sin(psi)) * (1 - ecc**2) ** 0.5
    qn0 = np.dot(sun, north)
    qe0 = np.dot(sun, east)
    _ypos = ypos * np.sqrt(1 - ecc**2)

    def geta(t):
        phi = (t - peri) / 365.25 * p % p
        psi = getpsi(phi)
        sun = (np.cos(psi) - ecc) * xpos + np.sin(psi) * _ypos
        # sun = np.matmul(xy, [np.cos(psi) - ecc, np.sin(psi)])
        qn = sun[0] * north[0] + sun[1] * north[1] + sun[2] * north[2]
        qe = sun[0] * east[0] + sun[1] * east[1] + sun[2] * east[2]

        qn -= qn0 + (qn2 - qn1) * (t - t0par) / 2
        qe -= qe0 + (qe2 - qe1) * (t - t0par) / 2

        return qn, qe

    return np.vectorize(geta)


def mag2flux(data):
    refmag = 18
    t = data[:, 0]
    if t[0] > 2450000:
        t -= 2450000
    mag = data[:, 1]
    err = data[:, 2]
    flux = 10 ** (0.4 * (refmag - mag))
    ferr = flux * err * 0.4 * np.log(10)
    return np.array([data[:, 0], flux, ferr])


def getb0p(z):
    '''
    return b0, b1, db0, db1
    '''
    if z < 0.001:
        return 2 * z, 2, -5.0 / 14.0 * z, -5.0 / 14.0
    if z > 10:
        return 1 + 1.0 / 8.0 / z**2, -1.0 / 4.0 / z**3, 1.0 / 40.0 / z**2, -1.0 / 20.0 / z**3
    else:
        return b0(z), b1(z), db0(z), db1(z)


vgetb0p = np.vectorize(getb0p)


def flux2mag(data, add=0):
    refmag = 18
    if len(data) == 1:
        return -2.5 * np.log10(data[0]) + refmag
    _data = data.copy()
    flux = data[1] + add
    ferr = data[2]
    mag = -2.5 * np.log10(flux) + refmag
    magerr = ferr / flux * 2.5 / np.log(10)
    _data[1] = mag
    _data[2] = magerr
    return _data


def gbad(_flux, chi2list):
    kntgood = len(chi2list)
    dof = len(chi2list) - 2
    mask = np.zeros(kntgood) > 1
    Flag = True
    while Flag:
        sigmax = np.sqrt(2 * np.log(kntgood / 3.0 / np.sqrt(3.14159 / 2)))
        sigmax = np.sqrt(2 * np.log(kntgood / sigmax / np.sqrt(3.14159 / 2)))
        # print("clip using sigma:", sigmax)
        chi2 = np.sum(chi2list[~mask])
        mask = chi2list > sigmax**2 * chi2 / kntgood
        rm = np.sum(mask)
        if kntgood + rm == dof + 2:
            Flag = False
        kntgood = dof + 2 - rm
    mask = np.where(mask == True)[0]
    output = ""
    for i in mask:
        output += f"{i:>10d}, # {chi2list[i]:10.1f} {_flux[0][i]:10.4f} {_flux[1][i]:10.4f} {_flux[2][i]:10.4f}\n"
    output = "bad = [\n" + output + "]"
    return output
