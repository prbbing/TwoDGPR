import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar

np.random.seed(42)


def estimate_abs_max(f, lim):
    a, b = float(lim[0]), float(lim[1])

    def neg_abs_f(x):
        return -abs(float(f(x)))

    res = minimize_scalar(neg_abs_f, bounds=(a, b), method="bounded")
    return max(abs(float(f(a))), abs(float(f(b))), -float(res.fun))


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


def dependent(num, xlim, ylim, xbin, ybin, cov, fx, fy, noisex=None, noisey=None):
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError("cov must be a 2x2 covariance matrix.")
    if not np.allclose(cov, cov.T):
        raise ValueError("cov must be symmetric.")
    if np.min(np.linalg.eigvalsh(cov)) < -1e-12:
        raise ValueError("cov must be positive semidefinite.")
    if noisex is None:
        noisex = np.random.normal(0, 1, num)
    if noisey is None:
        noisey = np.random.normal(0, 1, num)
    x = generate_data(num, xlim, noisex, fx, xbin)
    y = generate_data(num, ylim, noisey, fy, ybin)
    # 以下的逻辑使用了ai，采取高斯秩重拍的方法来生成相关数据，保持边缘分布不变，同时注入相关性。我不知道对不对，因为数学没学过
    # 这种方法的核心思想是：先生成独立的边缘数据，然后通过高斯秩重排（Gaussian copula）的方法注入相关性。具体步骤如下：
    # 1. 生成独立的边缘数据：首先，使用之前的方法生成独立的 x 和 y 数据，这些数据分别符合 fx 和 fy 定义的边缘分布。
    # 2. 计算相关系数：根据给定的协方差矩阵 cov，计算 x 和 y 之间的相关系数 rho。这个相关系数反映了 x 和 y 之间的线性关系强度。
    # 3. 生成高斯数据：使用 multivariate_normal 函数生成一组二维高斯数据 z，这些数据具有均值为 0 和协方差矩阵 [[1, rho], [rho, 1]]。这一步生成的 z 数据具有我们希望注入的相关性。
    # 4. 高斯秩重排：对 z 的两个维度分别进行排序，并获取排序后的索引 rx 和 ry。然后，根据这些索引对之前生成的独立边缘数据 x_vals 和 y_vals 进行重排，得到 x_dep 和 y_dep。
    # 这一步确保了 x_dep 和 y_dep 的边缘分布保持不变，同时注入了我们希望的相关性。
    # 取出你已经采好的边缘样本（保持 fx/fy 边缘）
    # 注意：这里的 x 和 y 是 generate_data 函数返回的元组，包含了数据和对应的 bins。我们只需要数据部分进行重排。
    # 因此，我们使用 x[0] 和 y[0] 来获取实际的样本数据，并将它们转换为 numpy 数组以便后续处理。
    # 通过这种方法，我们可以生成具有指定边缘分布和相关性的二维数据集。这种方法的优点是它保持了边缘分布不变，同时允许我们灵活地控制相关性。
    # 需要注意的是，这种方法生成的相关性是线性的，如果需要更复杂的相关结构，可能需要使用其他类型的 copula 或者非线性变换。
    # 总的来说，这种方法是一种有效的方式来生成具有特定边缘分布和相关性的二维数据，适用于许多统计模拟和数据分析的场景。
    # 从实践上来讲，这样的方法可以得到想要的结果，大致可以相信它是正确的，虽然我没有数学基础来证明它的正确性。
    x_vals = np.asarray(x[0], dtype=float)
    y_vals = np.asarray(y[0], dtype=float)

    # 用 cov 的相关系数做高斯秩重排（注入相关性，边缘不变）
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rho = np.clip(rho, -0.999, 0.999)

    z = np.random.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[1.0, rho], [rho, 1.0]],
        size=num,
    )

    rx = np.argsort(np.argsort(z[:, 0]))
    ry = np.argsort(np.argsort(z[:, 1]))

    x_dep = np.sort(x_vals)[rx]
    y_dep = np.sort(y_vals)[ry]

    xbins = np.linspace(xlim[0], xlim[1], xbin + 1)
    ybins = np.linspace(ylim[0], ylim[1], ybin + 1)

    plot_data(x_dep, y_dep, xbins, ybins, "Dependent Data")
    hist_data = get_hist_data(x_dep, y_dep, xbins, ybins)
    df = pd.DataFrame(hist_data, columns=["x", "y", "count"])
    return df


if __name__ == "__main__":
    # Example usage:
    num_points = 10000
    x_limits = (-0.5, 0.5)
    y_limits = (-0.5, 0.5)
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
        lambda x: 4 * np.exp(-((x) ** 2) / 0.05),
        lambda y: 4 * np.exp(-((y) ** 2) / 0.05),
        noise_x=np.zeros(num_points),
        noise_y=np.zeros(num_points),
    )
    df_dependent = dependent(
        num_points,
        x_limits,
        y_limits,
        x_bins,
        y_bins,
        covariance_matrix,
        lambda x: 4 * np.exp(-((x) ** 2) / 0.05),
        lambda y: 4 * np.exp(-((y) ** 2) / 0.05),
        noisex=np.zeros(num_points),
        noisey=np.zeros(num_points),
    )

    df_dependent.to_csv("dependent_data.csv", index=False)
    df_independent.to_csv("independent_data.csv", index=False)
