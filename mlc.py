import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import newton

plt.style.use("dark_background")

# s = [y(x)
#      v(x)]

F = lambda x, s: np.dot(np.array([[0,1],[1,0]]),s)

def yf(v0):
    sol = solve_ivp(F, [0, 2], [y0, v0])
    y, v = sol.y
    return y[-1] - 3.7622

y0 = 1. # Initial condition
x_eval = np.linspace(0, 2, 100) # x range of evaluation, taking 100 subdivisions
v0 = newton(yf, 8) # Initial guess = 8, we use Newton method as Shooting
print(v0)

sol = solve_ivp(F, [0, 2], [y0, v0], t_eval = x_eval)
plt.figure(figsize = (10, 8))
plt.plot(sol.t, sol.y[0]) # sol.t returns x values, sol.y returns solution values
plt.plot(2, 3.7622, "ro")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"root finding v={v0} m/s")
plt.show()
