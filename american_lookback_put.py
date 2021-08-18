import numpy as np
import matplotlib.pyplot as plt

def payoff(s):
    return np.max([1 - s, 0])

def left_boundary(r, t):
    return np.exp(-r*t)

def right_boundary():
    return 0

def jump(s, u, ds):
    one = int(1 / ds)
    # print(max(s * ds, 1))
    return max(s * ds, 1) * u[min(s, one)]

def get_An(sigma, r, n, dt):
    return 0.5 * (sigma**2 * n**2 - r * n) * dt

def get_Bn(sigma, r, n, dt):
    return (sigma**2 * n**2 + r) * dt

def get_Cn(sigma, r, n, dt):
    return 0.5 * (sigma**2 * n**2 + r * n) * dt

def get_S_grid(s, j, ds):
    return np.arange(0, s/j/ds + 1, dtype=int)

def get_sampling_times(u, l, step, dt):
    return np.arange(u/dt, l/dt, step/dt)[::-1]

def get_t_grid(T, dt):
    return np.arange(0, T/dt + 1)[::-1]

def american_lookback_put(dt, u, l, step, ds, smax, j, T, sigma, r):
    s_grid = get_S_grid(smax, j, ds)
    sp_grid = get_sampling_times(u, l, step, step * dt)
    t_grid = get_t_grid(T, step * dt)

    num_sp = 0

    A = np.array([get_An(sigma, r, n, dt) for n in s_grid])
    B = np.array([get_Bn(sigma, r, n, dt) for n in s_grid])
    C = np.array([get_Cn(sigma, r, n, dt) for n in s_grid])

    eps = np.power(10.0, -8)
    omega = 1
    domega = 0.05
    oldloops = 10000

    u = np.array([payoff(n * ds) for n in s_grid])

    for m in t_grid[1:]:
        g = np.array([payoff(n * ds) for n in s_grid])

        z = np.zeros(len(s_grid))
        z[0] = (2 - B[0]) * u[0] + C[0] * u[1]
        z[-1] = A[-1] * u[-2] + (2 - B[-1]) * u[-1]
        z[1:-1] = np.array([A[n] * u[n - 1] + (2 - B[n]) * u[n] + C[n] * u[n + 1] for n in s_grid[1:-1]])

        u[0] = left_boundary(r, m*dt)
        u[-1] = u[-2] # ???????

        l = psor_solver(u, z, g, A, B, C, s_grid, omega, eps)
        
        if l > oldloops:
            domega *= -1
        omega += domega
        oldloops = l

        if num_sp < len(sp_grid) and m == sp_grid[num_sp]:
            pre_jump_u = u[:]
            u = [jump(n, pre_jump_u, ds) for n in s_grid]
            num_sp += 1
    plt.plot(s_grid * ds, u)
    plt.grid()
    plt.show()
    return u 

def psor_solver(u, z, g, A, B, C, s_grid, omega, eps):
    l = 0
    error = 1
    while error > eps:
        error = 0
        for n in s_grid[1:-1]:
            y = (z[n] + A[n] * u[n - 1] + C[n] * u[n + 1]) / (2 + B[n])
            y = max(g[n], u[n] + omega * (y - u[n]))
            error += (u[n] - y)**2
            u[n] = y
        l += 1
    return l

def find_value(am, num, ds):
    return am[int(num / ds)]

if __name__ == '__main__':

    am_A = american_lookback_put(0.05, 0.5/12, 11.5/12, 1/12, 0.05, 250, 200, 1, 0.2, 0.1)
    #print(am)
    val_A = round(find_value(am=am_A, num=0.95, ds=0.05), 2)
    print("transformed = {} and untransformed = {} values of the option at one year to expiry when the asset price is 190 and the current maximum is 200 - case A".format(val_A, val_A * 200))

    # am_C = american_lookback_put(0.05, 1.5/12, 11.5/12, 2/12, 0.05, 250, 200, 1, 0.2, 0.1)
    # #print(am)
    # val_C = find_value(am=am_C, num=0.95, ds=0.05)
    # print(val_C, val_C * 200)

    am_B = american_lookback_put(0.05, 3.5/12, 11.5/12, 4/12, 0.05, 300, 200, 1, 0.2, 0.1)
    #print(am)
    val_B = round(find_value(am=am_B, num=0.95, ds=0.05), 2)
    print("transformed = {} and untransformed = {} values of the option at one year to expiry when the asset price is 190 and the current maximum is 200 - case B".format(val_B, val_B * 200))