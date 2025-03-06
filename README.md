import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk interpolasi linear
def linear_interpolation(x, y, x_new):
    return np.interp(x_new, x, y)

# Fungsi untuk metode polinom Newton
def newton_polynomial(x, y):
    n = len(x)
    coeff = np.zeros((n, n))
    coeff[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coeff[i][j] = (coeff[i + 1][j - 1] - coeff[i][j - 1]) / (x[i + j] - x[i])

    return coeff[0]

def evaluate_newton_polynomial(coeff, x, x_new):
    n = len(coeff) - 1
    result = coeff[n]
    for k in range(1, n + 1):
        term = coeff[n - k]
        for j in range(n - k, n):
            term *= (x_new - x[j])
        result += term
    return result

# Contoh penggunaan
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
x_new = np.linspace(1, 4, 100)

# Interpolasi Linear
y_linear = linear_interpolation(x, y, x_new)

# Metode Polinom Newton
coeff = newton_polynomial(x, y)
y_newton = evaluate_newton_polynomial(coeff, x, x_new)

# Plot hasil
plt.plot(x, y, 'o', label='Data Asli')
plt.plot(x_new, y_linear, label='Interpolasi Linear')
plt.plot(x_new, y_newton, label='Polinom Newton')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolasi Linear dan Metode Polinom Newton')
plt.grid()
plt.show()
