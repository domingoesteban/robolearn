from scipy.interpolate import CubicSpline
import numpy as np

#x = np.arange(10)
#y = np.sin(x)
#cs = CubicSpline(x, y)


class cubicSpline(object):
    def __init__(self, data, step_time):
        self.set_data(data)
        self.current_time = 0
        self.step_time = step_time

    def tick(self):
        self.current_time += self.step_time

    def output(self, time=None):

        if time is None:
            time = self.current_time

        x = time

        if time < 0:
            x = 0

        if time > self.x[-1]:
            x = self.x[-1]

        for i in xrange(len(self.x)):
            if x <= self.x[i + 1]:
                x_xi = x - self.x[i]
                print("Time: %d - %d = %d" % (x, self.x[i], x_xi))
                return self.coeffs[i, 0] + self.coeffs[i, 1]*x_xi + self.coeffs[i, 2]*np.power(x_xi, 2) + self.coeffs[i, 3]*np.power(x_xi, 3)

        return self.y[-1]

    def tick_and_output(self):
        self.tick()
        return self.output()

    def cyclic_tick_and_output(self, i):
        self.tick()

        if self.current_time > self.x[-1]:
            self.current_time = self.x[i]

        return self.output()

    def get_current_time(self):
        return self.current_time

    def get_step_time(self):
        return self.step_time

    def get_dta(self):
        return self.data

    def set_current_time(self, current_time):
        self.current_time = current_time

    def set_step_time(self, step_time):
        self.step_time = step_time

    def set_data(self, data):
        if data.shape[1] != 2:
            assert ValueError("The column of data is not 2!")

        A = 0.  # Initial dy
        B = 0.  # Ending dy

        self.data = data

        self.y = data[:, 1]

        n = len(self.y) - 1

        t = data[:, 0]

        self.h = t[-n:]

        self.x = np.zeros(n+1)

        for i in xrange(n):
            self.x[i+1] = self.x[i] + self.h[i]

        print('x')
        print(self.x)
        print("")

        print('h')
        print(self.h)
        print("")

        self.h_matrix = np.zeros([n+1, n+1])
        self.h_matrix[0, 0] = 2. * self.h[0]
        self.h_matrix[0, 1] = self.h[0]
        self.h_matrix[n, n-1] = self.h[n-1]
        self.h_matrix[n, n] = 2.*self.h[n-1]

        for i in xrange(1, n):
            self.h_matrix[i, i-1] = self.h[i-1]
            self.h_matrix[i, i] = 2. * (self.h[i-1] + self.h[i])
            self.h_matrix[i, i+1] = self.h[i]
        print(self.h_matrix)

        print('y')
        print(self.y)
        print("")

        self.yh_vector = np.zeros(n+1)
        self.yh_vector[0] = 6. * ((self.y[1] - self.y[0]) / self.h[0] - A)
        self.yh_vector[n] = 6. * (B - (self.y[n] - self.y[n-1]) / self.h[n-1])
        print((self.y[n] - self.y[n-1]) / self.h[n-1])
        for i in xrange(1, n):
            self.yh_vector[i] = 6. * ((self.y[i+1] - self.y[i]) / self.h[i] - (self.y[i] - self.y[i-1]) / self.h[i-1])
        print('yh_vector')
        print(self.yh_vector)

        #self.m = np.linalg.inv(self.h_matrix).dot(self.yh_vector)
        self.m = np.linalg.solve(self.h_matrix, self.yh_vector)

        self.coeffs = np.zeros([n, 4])
        for i in xrange(n):
            self.coeffs[i, 0] = self.y[i]  # a(i)
            self.coeffs[i, 1] = (self.y[i+1] - self.y[i])/self.h[i] - self.h[i]*self.m[i]/2 - self.h[i]*(self.m[i+1] - self.m[i])/6.  # b(i)
            self.coeffs[i, 2] = self.m[i]/2.  # c(i)
            self.coeffs[i, 3] = (self.m[i+1] - self.m[i]) / (6. * self.h[i])  # d(i)


data = np.array([[0, 3], [10, 9], [10, 8]], dtype=np.float)#, ])
print(data)
step_time = 1

interpolator = cubicSpline(data, step_time)

for i in xrange(30):
    print("%d, %f" % (i, interpolator.tick_and_output()))
    print("")

