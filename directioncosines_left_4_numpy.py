# Direction Cosines Left Angular Velocity 4 Program in Python.
# Calculates the Numerical Approximation of the Motion of a 3-D Gyroscopic Pendulum
# Using 4th-Order Runge-Kutta Method.

from numpy import array, cos, float64, linspace, pi, sin, sqrt, zeros

#
# zyz-Convention Euler Angles Representation of the Proper Orthogonal Linear Transformation
#


def oxx(qyaw, qpitch, qroll):
    f = cos(qyaw) * cos(qpitch) * cos(qroll) - sin(qyaw) * sin(qroll)
    return f


def oyx(qyaw, qpitch, qroll):
    f = sin(qyaw) * cos(qpitch) * cos(qroll) + cos(qyaw) * sin(qroll)
    return f


def ozx(qpitch, qroll):
    f = - sin(qpitch) * cos(qroll)
    return f


def oxy(qyaw, qpitch, qroll):
    f = - cos(qyaw) * cos(qpitch) * sin(qroll) - sin(qyaw) * cos(qroll)
    return f


def oyy(qyaw, qpitch, qroll):
    f = - sin(qyaw) * cos(qpitch) * sin(qroll) + cos(qyaw) * cos(qroll)
    return f


def ozy(qpitch, qroll):
    f = sin(qpitch) * sin(qroll)
    return f


def oxz(qyaw, qpitch):
    f = cos(qyaw) * sin(qpitch)
    return f


def oyz(qyaw, qpitch):
    f = sin(qyaw) * sin(qpitch)
    return f


def ozz(qpitch):
    f = cos(qpitch)
    return f


#
# zyz-Convention Euler Angles Representation of the Left Angular Velocity
#


def bomegax(qyaw, qpitch, dqpitch, dqroll):
    f = dqroll * sin(qpitch) * cos(qyaw) - dqpitch * sin(qyaw)
    return f


def bomegay(qyaw, qpitch, dqpitch, dqroll):
    f = dqroll * sin(qpitch) * sin(qyaw) + dqpitch * cos(qyaw)
    return f


def bomegaz(qpitch, dqyaw, dqroll):
    f = dqroll * cos(qpitch) + dqyaw
    return f


def main():
    ns = 700000
    nt = 700
    a = 0.25
    h = 0.1
    l = 1.0
    ipum = array([0.0, (a ** 2 + h ** 2 / 3.0) / 4.0 + l ** 2, (a ** 2 + h ** 2 / 3.0) / 4.0 + l ** 2, a ** 2 / 2.0],
                 dtype='float64')
    gamma = array([0.0, (ipum[2] - ipum[3]) / ipum[1], (ipum[3] - ipum[1]) / ipum[2], (ipum[1] - ipum[2]) / ipum[3]],
                 dtype='float64')
    beta = array([0.0, l / ipum[1], l / ipum[2], l / ipum[3]], dtype='float64')
    bg = array([0.0, 0.0, 0.0, -9.8], dtype='float64')
    time = linspace(0.0, 7.0, ns+1, dtype='float64')
    yaw = 0.0 * float64(pi)
    pitch = 11.0 * float64(pi) / 24.0
    roll = 0.0 * float64(pi)
    o = array([[0.0, 0.0, 0.0, 0.0],
         [0.0, oxx(yaw, pitch, roll), oxy(yaw, pitch, roll), oxz(yaw, pitch)],
         [0.0, oyx(yaw, pitch, roll), oyy(yaw, pitch, roll), oyz(yaw, pitch)],
         [0.0, ozx(pitch, roll), ozy(pitch, roll), ozz(pitch)]], dtype='float64')
    bi = array([0.0, o[1][1], o[2][1], o[3][1]], dtype='float64')
    bj = array([0.0, o[1][2], o[2][2], o[3][2]], dtype='float64')
    bk = array([0.0, o[1][3], o[2][3], o[3][3]], dtype='float64')
    bgbi = bg[1] * bi[1] + bg[2] * bi[2] + bg[3] * bi[3]
    bgbj = bg[1] * bj[1] + bg[2] * bj[2] + bg[3] * bj[3]
    bgbk = bg[1] * bk[1] + bg[2] * bk[2] + bg[3] * bk[3]
    dyaw = 0.0 * float64(pi)
    dpitch = 0.0 * float64(pi)
    droll = 100.0 * float64(pi)
    bomega = array([0.0, bomegax(yaw, pitch, dpitch, droll), bomegay(yaw, pitch, dpitch, droll), bomegaz(pitch, dyaw, droll)], dtype='float64')
    bobi = bomega[1] * bi[1] + bomega[2] * bi[2] + bomega[3] * bi[3]
    bobj = bomega[1] * bj[1] + bomega[2] * bj[2] + bomega[3] * bj[3]
    bobk = bomega[1] * bk[1] + bomega[2] * bk[2] + bomega[3] * bk[3]
    dbomega = array([0.0,
               gamma[1] * bobj * bobk * bi[1] + gamma[2] * bobk * bobi * bj[1] + gamma[3] * bobi * bobj * bk[1]
               - beta[1] * bgbj * bi[1] + beta[2] * bgbi * bj[1],
               gamma[1] * bobj * bobk * bi[2] + gamma[2] * bobk * bobi * bj[2] + gamma[3] * bobi * bobj * bk[2]
               - beta[1] * bgbj * bi[2] + beta[2] * bgbi * bj[2],
               gamma[1] * bobj * bobk * bi[3] + gamma[2] * bobk * bobi * bj[3] + gamma[3] * bobi * bobj * bk[3]
               - beta[1] * bgbj * bi[3] + beta[2] * bgbi * bj[3]], dtype='float64')
    oj = o
    bomegaj = bomega
    ot = zeros((nt + 1, 4, 4))
    ot[0][1][1] = o[1][1]
    ot[0][2][1] = o[2][1]
    ot[0][3][1] = o[3][1]
    ot[0][1][2] = o[1][2]
    ot[0][2][2] = o[2][2]
    ot[0][3][2] = o[3][2]
    ot[0][1][3] = o[1][3]
    ot[0][2][3] = o[2][3]
    ot[0][3][3] = o[3][3]
    bomegat = zeros((nt + 1, 4))
    bomegat[0][1] = bomega[1]
    bomegat[0][2] = bomega[2]
    bomegat[0][3] = bomega[3]
    bit = zeros((nt + 1, 4))
    bit[0][1] = bi[1]
    bit[0][2] = bi[2]
    bit[0][3] = bi[3]
    bjt = zeros((nt + 1, 4))
    bjt[0][1] = bj[1]
    bjt[0][2] = bj[2]
    bjt[0][3] = bj[3]
    bkt = zeros((nt + 1, 4))
    bkt[0][1] = bk[1]
    bkt[0][2] = bk[2]
    bkt[0][3] = bk[3]
    for k in range(1, nt + 1, 1):
        for j in range(1, ns // nt + 1, 1):
            nj = ns // nt * (k - 1) + j - 1
            deltat = time[nj + 1] - time[nj]
            fo1 = array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, bomega[2] * o[3][1] - bomega[3] * o[2][1],
                    bomega[2] * o[3][2] - bomega[3] * o[2][2],
                    bomega[2] * o[3][3] - bomega[3] * o[2][3]],
                   [0.0, bomega[3] * o[1][1] - bomega[1] * o[3][1],
                    bomega[3] * o[1][2] - bomega[1] * o[3][2],
                    bomega[3] * o[1][3] - bomega[1] * o[3][3]],
                   [0.0, bomega[1] * o[2][1] - bomega[2] * o[1][1],
                    bomega[1] * o[2][2] - bomega[2] * o[1][2],
                    bomega[1] * o[2][3] - bomega[2] * o[1][3]]], dtype='float64')
            fomega1 = dbomega
            o = oj + deltat * fo1 / 2.0
            bomega = bomegaj + deltat * fomega1 / 2.0
            bi = array([0.0, o[1][1], o[2][1], o[3][1]], dtype='float64')
            bj = array([0.0, o[1][2], o[2][2], o[3][2]], dtype='float64')
            bk = array([0.0, o[1][3], o[2][3], o[3][3]], dtype='float64')
            bgbi = bg[1] * bi[1] + bg[2] * bi[2] + bg[3] * bi[3]
            bgbj = bg[1] * bj[1] + bg[2] * bj[2] + bg[3] * bj[3]
            bgbk = bg[1] * bk[1] + bg[2] * bk[2] + bg[3] * bk[3]
            bobi = bomega[1] * bi[1] + bomega[2] * bi[2] + bomega[3] * bi[3]
            bobj = bomega[1] * bj[1] + bomega[2] * bj[2] + bomega[3] * bj[3]
            bobk = bomega[1] * bk[1] + bomega[2] * bk[2] + bomega[3] * bk[3]
            dbomega = array([0.0,
                       gamma[1] * bobj * bobk * bi[1] + gamma[2] * bobk * bobi * bj[1] + gamma[3] * bobi * bobj * bk[1]
                       - beta[1] * bgbj * bi[1] + beta[2] * bgbi * bj[1],
                       gamma[1] * bobj * bobk * bi[2] + gamma[2] * bobk * bobi * bj[2] + gamma[3] * bobi * bobj * bk[2]
                       - beta[1] * bgbj * bi[2] + beta[2] * bgbi * bj[2],
                       gamma[1] * bobj * bobk * bi[3] + gamma[2] * bobk * bobi * bj[3] + gamma[3] * bobi * bobj * bk[3]
                       - beta[1] * bgbj * bi[3] + beta[2] * bgbi * bj[3]], dtype='float64')
            fo2 = array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, bomega[2] * o[3][1] - bomega[3] * o[2][1],
                    bomega[2] * o[3][2] - bomega[3] * o[2][2],
                    bomega[2] * o[3][3] - bomega[3] * o[2][3]],
                   [0.0, bomega[3] * o[1][1] - bomega[1] * o[3][1],
                    bomega[3] * o[1][2] - bomega[1] * o[3][2],
                    bomega[3] * o[1][3] - bomega[1] * o[3][3]],
                   [0.0, bomega[1] * o[2][1] - bomega[2] * o[1][1],
                    bomega[1] * o[2][2] - bomega[2] * o[1][2],
                    bomega[1] * o[2][3] - bomega[2] * o[1][3]]], dtype='float64')
            fomega2 = dbomega
            o = oj + deltat * fo2 / 2.0
            bomega = bomegaj + deltat * fomega2 / 2.0
            bi = array([0.0, o[1][1], o[2][1], o[3][1]], dtype='float64')
            bj = array([0.0, o[1][2], o[2][2], o[3][2]], dtype='float64')
            bk = array([0.0, o[1][3], o[2][3], o[3][3]], dtype='float64')
            bgbi = bg[1] * bi[1] + bg[2] * bi[2] + bg[3] * bi[3]
            bgbj = bg[1] * bj[1] + bg[2] * bj[2] + bg[3] * bj[3]
            bgbk = bg[1] * bk[1] + bg[2] * bk[2] + bg[3] * bk[3]
            bobi = bomega[1] * bi[1] + bomega[2] * bi[2] + bomega[3] * bi[3]
            bobj = bomega[1] * bj[1] + bomega[2] * bj[2] + bomega[3] * bj[3]
            bobk = bomega[1] * bk[1] + bomega[2] * bk[2] + bomega[3] * bk[3]
            dbomega = array([0.0,
                       gamma[1] * bobj * bobk * bi[1] + gamma[2] * bobk * bobi * bj[1] + gamma[3] * bobi * bobj * bk[1]
                       - beta[1] * bgbj * bi[1] + beta[2] * bgbi * bj[1],
                       gamma[1] * bobj * bobk * bi[2] + gamma[2] * bobk * bobi * bj[2] + gamma[3] * bobi * bobj * bk[2]
                       - beta[1] * bgbj * bi[2] + beta[2] * bgbi * bj[2],
                       gamma[1] * bobj * bobk * bi[3] + gamma[2] * bobk * bobi * bj[3] + gamma[3] * bobi * bobj * bk[3]
                       - beta[1] * bgbj * bi[3] + beta[2] * bgbi * bj[3]], dtype='float64')
            fo3 = array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, bomega[2] * o[3][1] - bomega[3] * o[2][1],
                    bomega[2] * o[3][2] - bomega[3] * o[2][2],
                    bomega[2] * o[3][3] - bomega[3] * o[2][3]],
                   [0.0, bomega[3] * o[1][1] - bomega[1] * o[3][1],
                    bomega[3] * o[1][2] - bomega[1] * o[3][2],
                    bomega[3] * o[1][3] - bomega[1] * o[3][3]],
                   [0.0, bomega[1] * o[2][1] - bomega[2] * o[1][1],
                    bomega[1] * o[2][2] - bomega[2] * o[1][2],
                    bomega[1] * o[2][3] - bomega[2] * o[1][3]]], dtype='float64')
            fomega3 = dbomega
            o = oj + deltat * fo3
            bomega = bomegaj + deltat * fomega3
            bi = array([0.0, o[1][1], o[2][1], o[3][1]], dtype='float64')
            bj = array([0.0, o[1][2], o[2][2], o[3][2]], dtype='float64')
            bk = array([0.0, o[1][3], o[2][3], o[3][3]], dtype='float64')
            bgbi = bg[1] * bi[1] + bg[2] * bi[2] + bg[3] * bi[3]
            bgbj = bg[1] * bj[1] + bg[2] * bj[2] + bg[3] * bj[3]
            bgbk = bg[1] * bk[1] + bg[2] * bk[2] + bg[3] * bk[3]
            bobi = bomega[1] * bi[1] + bomega[2] * bi[2] + bomega[3] * bi[3]
            bobj = bomega[1] * bj[1] + bomega[2] * bj[2] + bomega[3] * bj[3]
            bobk = bomega[1] * bk[1] + bomega[2] * bk[2] + bomega[3] * bk[3]
            dbomega = array([0.0,
                       gamma[1] * bobj * bobk * bi[1] + gamma[2] * bobk * bobi * bj[1] + gamma[3] * bobi * bobj * bk[1]
                       - beta[1] * bgbj * bi[1] + beta[2] * bgbi * bj[1],
                       gamma[1] * bobj * bobk * bi[2] + gamma[2] * bobk * bobi * bj[2] + gamma[3] * bobi * bobj * bk[2]
                       - beta[1] * bgbj * bi[2] + beta[2] * bgbi * bj[2],
                       gamma[1] * bobj * bobk * bi[3] + gamma[2] * bobk * bobi * bj[3] + gamma[3] * bobi * bobj * bk[3]
                       - beta[1] * bgbj * bi[3] + beta[2] * bgbi * bj[3]], dtype='float64')
            fo4 = array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, bomega[2] * o[3][1] - bomega[3] * o[2][1],
                    bomega[2] * o[3][2] - bomega[3] * o[2][2],
                    bomega[2] * o[3][3] - bomega[3] * o[2][3]],
                   [0.0, bomega[3] * o[1][1] - bomega[1] * o[3][1],
                    bomega[3] * o[1][2] - bomega[1] * o[3][2],
                    bomega[3] * o[1][3] - bomega[1] * o[3][3]],
                   [0.0, bomega[1] * o[2][1] - bomega[2] * o[1][1],
                    bomega[1] * o[2][2] - bomega[2] * o[1][2],
                    bomega[1] * o[2][3] - bomega[2] * o[1][3]]], dtype='float64')
            fomega4 = dbomega
            o = oj + deltat * (fo1 + 2.0 * fo2 + 2.0 * fo3 + fo4) / 6.0
            bomega = bomegaj + deltat * (fomega1 + 2.0 * fomega2 + 2.0 * fomega3 + fomega4) / 6.0
            bi = array([0.0, o[1][1], o[2][1], o[3][1]], dtype='float64')
            bj = array([0.0, o[1][2], o[2][2], o[3][2]], dtype='float64')
            bk = array([0.0, o[1][3], o[2][3], o[3][3]], dtype='float64')
            bgbi = bg[1] * bi[1] + bg[2] * bi[2] + bg[3] * bi[3]
            bgbj = bg[1] * bj[1] + bg[2] * bj[2] + bg[3] * bj[3]
            bgbk = bg[1] * bk[1] + bg[2] * bk[2] + bg[3] * bk[3]
            bobi = bomega[1] * bi[1] + bomega[2] * bi[2] + bomega[3] * bi[3]
            bobj = bomega[1] * bj[1] + bomega[2] * bj[2] + bomega[3] * bj[3]
            bobk = bomega[1] * bk[1] + bomega[2] * bk[2] + bomega[3] * bk[3]
            dbomega = array([0.0,
                       gamma[1] * bobj * bobk * bi[1] + gamma[2] * bobk * bobi * bj[1] + gamma[3] * bobi * bobj * bk[1]
                       - beta[1] * bgbj * bi[1] + beta[2] * bgbi * bj[1],
                       gamma[1] * bobj * bobk * bi[2] + gamma[2] * bobk * bobi * bj[2] + gamma[3] * bobi * bobj * bk[2]
                       - beta[1] * bgbj * bi[2] + beta[2] * bgbi * bj[2],
                       gamma[1] * bobj * bobk * bi[3] + gamma[2] * bobk * bobi * bj[3] + gamma[3] * bobi * bobj * bk[3]
                       - beta[1] * bgbj * bi[3] + beta[2] * bgbi * bj[3]], dtype='float64')
            oj = o
            bomegaj = bomega
        ot[k][1][1] = o[1][1]
        ot[k][2][1] = o[2][1]
        ot[k][3][1] = o[3][1]
        ot[k][1][2] = o[1][2]
        ot[k][2][2] = o[2][2]
        ot[k][3][2] = o[3][2]
        ot[k][1][3] = o[1][3]
        ot[k][2][3] = o[2][3]
        ot[k][3][3] = o[3][3]
        bomegat[k][1] = bomega[1]
        bomegat[k][2] = bomega[2]
        bomegat[k][3] = bomega[3]
        bit[k][1] = bi[1]
        bit[k][2] = bi[2]
        bit[k][3] = bi[3]
        bjt[k][1] = bj[1]
        bjt[k][2] = bj[2]
        bjt[k][3] = bj[3]
        bkt[k][1] = bk[1]
        bkt[k][2] = bk[2]
        bkt[k][3] = bk[3]
        print("%7u%3c%- 22.15e" % (nj + 1, ' ', time[nj + 1]))
    unit1 = open("dc_l_4_primary.out", "w")
    unit2 = open("dc_l_4_secondary.out", "w")
    unit1.write("%9c%4s%20c%7s%18c%7s%18c%7s%20c%3s%22c%3s%22c%3s%22c%3s%22c%3s%22c%3s%22c%3s%22c%3s%22c%3s\n" %
                (' ', "time", ' ', "bomegax", ' ', "bomegay", ' ', "bomegaz", ' ', "bix", ' ', "biy", ' ', "biz",
                 ' ', "bjx", ' ', "bjy", ' ', "bjz", ' ', "bkx", ' ', "bky", ' ', "bkz"))
    unit2.write("%9c%4s%21c%5s%20c%5s%20c%5s%20c%4s%21c%4s%21c%4s%21c%4s\n" %
                (' ', "time", ' ', "bimag", ' ', "bjmag", ' ', "bkmag", ' ', "bibj", ' ', "bkbi", ' ', "bjbk", ' ',
                 "epum"))
    for k in range(0, nt + 1, 1):
        bimag = sqrt(bit[k][1] ** 2 + bit[k][2] ** 2 + bit[k][3] ** 2)
        bjmag = sqrt(bjt[k][1] ** 2 + bjt[k][2] ** 2 + bjt[k][3] ** 2)
        bkmag = sqrt(bkt[k][1] ** 2 + bkt[k][2] ** 2 + bkt[k][3] ** 2)
        bibj = bit[k][1] * bjt[k][1] + bit[k][2] * bjt[k][2] + bit[k][3] * bjt[k][3]
        bkbi = bkt[k][1] * bit[k][1] + bkt[k][2] * bit[k][2] + bkt[k][3] * bit[k][3]
        bjbk = bjt[k][1] * bkt[k][1] + bjt[k][2] * bkt[k][2] + bjt[k][3] * bkt[k][3]
        bobi = bomegat[k][1] * bit[k][1] + bomegat[k][2] * bit[k][2] + bomegat[k][3] * bit[k][3]
        bobj = bomegat[k][1] * bjt[k][1] + bomegat[k][2] * bjt[k][2] + bomegat[k][3] * bjt[k][3]
        bobk = bomegat[k][1] * bkt[k][1] + bomegat[k][2] * bkt[k][2] + bomegat[k][3] * bkt[k][3]
        bgbk = bg[1] * bkt[k][1] + bg[2] * bkt[k][2] + bg[3] * bkt[k][3]
        epum = (ipum[1] * bobi ** 2 + ipum[2] * bobj ** 2 + ipum[3] * bobk ** 2) / 2.0 - l * bgbk
        unit1.write(
         "%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c"
         "%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c%- 22.15e\n"
         % (time[ns // nt * k], ' ',
            bomegat[k][1], ' ', bomegat[k][2], ' ', bomegat[k][3], ' ',
            bit[k][1], ' ', bit[k][2], ' ', bit[k][3], ' ',
            bjt[k][1], ' ', bjt[k][2], ' ', bjt[k][3], ' ',
            bkt[k][1], ' ', bkt[k][2], ' ', bkt[k][3]))
        unit2.write(
         "%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c%- 22.15e%3c%- 22.15e\n"
         % (time[ns // nt * k], ' ', bimag, ' ', bjmag, ' ', bkmag, ' ', bibj, ' ', bkbi, ' ', bjbk, ' ', epum))
    unit1.close()
    unit2.close()


if __name__ == '__main__':
    main()
    exit(0)
