import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar, minimize

np.random.seed(42)


def estimate_abs_max(f, lim):
    a, b = float(lim[0]), float(lim[1])

    def neg_abs_f(x):
        return -abs(float(f(x)))

    res = minimize_scalar(neg_abs_f, bounds=(a, b), method="bounded")
    return max(abs(float(f(a))), abs(float(f(b))), -float(res.fun))


def estimate_abs_max_2d(fxy, xlim, ylim):
    def neg_abs_f(v):
        return -abs(float(fxy(v[0], v[1])))

    # Simple grid search for a good initial guess
    x0_vals = np.linspace(xlim[0], xlim[1], 10)
    y0_vals = np.linspace(ylim[0], ylim[1], 10)
    best_v = [x0_vals[0], y0_vals[0]]
    min_val = 0
    for vx in x0_vals:
        for vy in y0_vals:
            val = neg_abs_f([vx, vy])
            if val < min_val:
                min_val = val
                best_v = [vx, vy]

    res = minimize(neg_abs_f, x0=best_v, bounds=(xlim, ylim))
    corners = [
        abs(float(fxy(xlim[0], ylim[0]))),
        abs(float(fxy(xlim[0], ylim[1]))),
        abs(float(fxy(xlim[1], ylim[0]))),
        abs(float(fxy(xlim[1], ylim[1]))),
    ]
    return max(max(corners), -float(res.fun))


def generate_data(points, lim, noise, f, bin):
    y = []

    f_max = estimate_abs_max(f, lim)
    M = 1.1 * f_max

    while len(y) < points:
        x = np.random.uniform(lim[0], lim[1], points)
        p_x = np.abs(np.asarray(f(x), dtype=float)) / M
        p_x = np.clip(p_x, 0.0, 1.0)
        random_values = np.random.uniform(0, 1, points)
        accepted = random_values < p_x
        y.extend(x[accepted].tolist())
    y = y[:points]
    y = np.array(y) + noise[:points]
    bins = np.linspace(lim[0], lim[1], bin + 1)
    return y, bins


def plot_data(x, y, xbins, ybins, name):
    plt.hist2d(x, y, bins=[xbins, ybins], cmap="Blues")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Histogram of Generated Data - " + name)
    plt.savefig(name.replace(" ", "_").lower() + ".png")
    plt.show()


def get_hist_data(x, y, xbins, ybins):
    hist, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
    des_x = (xedges[:-1] + xedges[1:]) / 2
    des_y = (yedges[:-1] + yedges[1:]) / 2
    out_data = []
    for i in range(len(des_x)):
        for j in range(len(des_y)):
            out_data.append([des_x[i], des_y[j], hist[i, j]])
    return out_data


def independent(
    num_x, num_y, xlim, ylim, xbin, ybin, fx, fy, noise_x=None, noise_y=None
):
    if noise_x is None:
        noise_x = np.random.normal(0, 1, num_x)
    if noise_y is None:
        noise_y = np.random.normal(0, 1, num_y)
    if num_x != num_y:
        raise ValueError("num_x and num_y must be the same for independent data.")
    x = generate_data(num_x, xlim, noise_x, fx, xbin)
    y = generate_data(num_y, ylim, noise_y, fy, ybin)
    plot_data(x[0], y[0], x[1], y[1], "Independent Data")
    hist_data = get_hist_data(x[0], y[0], x[1], y[1])
    df = pd.DataFrame(hist_data, columns=["x", "y", "count"])
    return df


def dependent(num, xlim, ylim, xbin, ybin, fxy, noise_x=None, noise_y=None):
    if noise_x is None:
        noise_x = np.random.normal(0, 1, num)
    if noise_y is None:
        noise_y = np.random.normal(0, 1, num)

    f_max = estimate_abs_max_2d(fxy, xlim, ylim)
    M = 1.1 * f_max

    x_res = []
    y_res = []

    while len(x_res) < num:
        x = np.random.uniform(xlim[0], xlim[1], num)
        y = np.random.uniform(ylim[0], ylim[1], num)
        p_xy = np.abs(np.asarray(fxy(x, y), dtype=float)) / M
        p_xy = np.clip(p_xy, 0.0, 1.0)
        random_values = np.random.uniform(0, 1, num)
        accepted = random_values < p_xy
        x_res.extend(x[accepted].tolist())
        y_res.extend(y[accepted].tolist())

    x_res = np.array(x_res[:num]) + noise_x[:num]
    y_res = np.array(y_res[:num]) + noise_y[:num]

    xbins = np.linspace(xlim[0], xlim[1], xbin + 1)
    ybins = np.linspace(ylim[0], ylim[1], ybin + 1)

    plot_data(x_res, y_res, xbins, ybins, "Dependent Data")
    hist_data = get_hist_data(x_res, y_res, xbins, ybins)
    df = pd.DataFrame(hist_data, columns=["x", "y", "count"])
    return df


if __name__ == "__main__":
    # Example usage:
    num_points = 10000
    x_limits = (-10, 10)
    y_limits = (-10, 10)
    x_bins = 30
    y_bins = 30
    covariance_matrix = [[1, 0.8], [0.8, 1]]

    df_independent = independent(
        num_points,
        num_points,
        x_limits,
        y_limits,
        x_bins,
        y_bins,
        lambda x: 4 * np.exp(-0.01 * (x - 5)),
        lambda y: 4 * np.exp(-0.01 * (y - 5)),
        noise_x=np.zeros(num_points),
        noise_y=np.zeros(num_points),
    )
    df_dependent = dependent(
        num_points,
        x_limits,
        y_limits,
        x_bins,
        y_bins,
        lambda x, y: 4
        * np.exp(-0.01 * (x - 5))
        * 4
        * np.exp(-0.01 * (y - 5))
        * (1 + 0.5 * np.sin(x) * np.sin(y)),
        noise_x=np.zeros(num_points),
        noise_y=np.zeros(num_points),
    )

    df_dependent.to_csv("dependent_data1.csv", index=False)
    df_independent.to_csv("independent_data1.csv", index=False)
