import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class UR5eDH(DHRobot):
    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym

            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi

            zero = 0.0

        deg = pi / 180

        # robot length values (metres)
        a = [0, -0.42500, -0.3922, 0, 0, 0]
        d = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
        alpha = [pi / 2, zero, zero, pi / 2, -pi / 2, zero]

        # mass and center of mass
        mass = [3.7000, 8.058, 2.846, 1.37, 1.3, 0.365]
        center_of_mass = [
            [0, -0.02561, 0.00193],
            [0.2125, 0, 0.11336],
            [0.15, 0, 0.0265],
            [0, -0.0018, 0.01634],
            [0, 0.0018, 0.01634],
            [0, 0, -0.001159],
        ]
        # inertia tensor
        inertia = [
            np.zeros((3, 3)),  # Link 1
            np.zeros((3, 3)),  # Link 2
            np.zeros((3, 3)),  # Link 3
            np.zeros((3, 3)),  # Link 4
            np.zeros((3, 3)),  # Link 5
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0.0001]]),  # Link 6 (non-zero Izz)
        ]

        links = []
        for j in range(6):
            link = RevoluteDH(
                d=d[j], a=a[j], alpha=alpha[j], m=mass[j], r=center_of_mass[j], G=1, I=inertia[j]
            )
            links.append(link)

        super().__init__(
            links,
            name="UR5e",
            manufacturer="Universal Robotics",
            keywords=("dynamics", "symbolic"),
            symbolic=symbolic,
        )

        # Named configurations
        # self.qr = np.radians([180, 0, 0, 0, 90, 0])
        # self.qz = np.zeros(6)
        # Default joint configuration (q)
        self.q = np.zeros(6)
        self.q_HOME =[0.701172053107018, 0.184272460738082, 0.1721568294843568,
                       -1.7318488600590023, 0.686830145115122, -1.731258978679887]
        # self.addconfiguration("qr", self.qr)
        # self.addconfiguration("qz", self.qz)
        self.addconfiguration("q_HOME", self.q_HOME)


if __name__ == "__main__":
    ur5e = UR5eDH(symbolic=False)
    print(ur5e)
