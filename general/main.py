import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import newton


def yf(v0):
    sol = solve_ivp(F, [x0, xf], [y0, v0])
    y, v = sol.y
    return y[-1] - y2


# ----------------------- Equations to solve -----------------------
# s = [y(x)
#      v(x)]
F = lambda x, s: np.dot(np.array([[0, 1], [-3 / x ** 2, 3 / x + (24 * x ** 3) / s[1]]]), s)


def aSol(r):  # analytical solution
    return 3 * r ** 5 - 15 * r ** 3 + 12 * r


# ----------------------- Boundary conditions -----------------------
x0 = 1.
y0 = 0.  # Initial condition
xf = 2.
y2 = 0.
x_eval = np.linspace(x0, xf, 100)  # x range of evaluation, taking 100 subdivisions
v0 = newton(yf, 8)  # Initial guess = 8, we use Newton method as Shooting
print(v0)

# ----------------------- Solving equations with RK45 -----------------------
sol = solve_ivp(F, [x0, xf], [y0, v0], t_eval=x_eval)  # This uses RK45 to numerically integrate

# ----------------------- Plotting -----------------------
plt.style.use("dark_background")
plt.figure(figsize=(10, 8))
plt.plot(sol.t, sol.y[0], 'bo')  # sol.t returns x values, sol.y returns solution values
plt.plot(xf, y2, "ro")
plt.plot(x_eval, aSol(x_eval), 'w')
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"root finding v={v0} m/s")
plt.show()
