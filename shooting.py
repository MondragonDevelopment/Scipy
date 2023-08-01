import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import newton

plt.style.use("dark_background")

F = lambda t, s: np.dot(np.array([[0,1],[0,-9.8/s[1]]]),s)
y0 = 0
t_eval = np.linspace(0, 5, 100)

def yf(v0):
    sol = solve_ivp(F, [0, 5], [y0, v0])
    y, v = sol.y
    return y[-1] - 50

v0 = newton(yf, 10)
print(v0)

sol = solve_ivp(F, [0, 5], [y0, v0], t_eval = t_eval)
plt.figure(figsize = (10, 8))
plt.plot(sol.t, sol.y[0])
plt.plot(5, 50, "ro")
plt.xlabel("time (s)")
plt.ylabel("altitude (m)")
plt.title(f"root finding v={v0} m/s")
plt.show()
