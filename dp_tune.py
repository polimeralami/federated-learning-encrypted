import subprocess
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import copy
import math
import re
import os


def compute_epsilon(C, sigma, delta=1e-3):
    # Correct DP epsilon formula using Gaussian mechanism
    return math.sqrt(2 * math.log(1.25 / delta)) * C / sigma


C_list = np.array([0.1, 0.3, 0.5, 0.9])
sigma_list = np.array([0.1, 0.2, 0.3, 0.4])

train_acc_map = {C: dict.fromkeys(sigma_list) for C in C_list}
test_acc_map = copy.deepcopy(train_acc_map)
loss_map = copy.deepcopy(train_acc_map)
epsilon_map = copy.deepcopy(train_acc_map)

for C in C_list:
    for sigma in sigma_list:
        print(f'Running DP with C: {C}\tsigma: {sigma}')
        epsilon_map[C][sigma] = compute_epsilon(C, sigma)

        cmd = [
            "python", "main.py", "--mode", "DP", "--gpu", "0", "--no-plot", "--epoch", "2",
            "--C", str(C), "--sigma", str(sigma)
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        out, _ = proc.communicate()
        out = out.decode('utf-8')
        print('out:', out)

        # Expecting output like:
        # Train Accuracy: 85.2, Test Accuracy: 83.1, Loss: 0.421
        try:
            train_acc = float(re.search(r"Train Accuracy:\s*([0-9.]+)", out).group(1))
            test_acc = float(re.search(r"Test Accuracy:\s*([0-9.]+)", out).group(1))
            loss = float(re.search(r"Loss:\s*([0-9.]+)", out).group(1))
        except AttributeError:
            print(f"[!] Failed to parse output for C={C}, sigma={sigma}")
            train_acc = test_acc = loss = 0.0  # fallback or skip

        print(f"Parsed: Train={train_acc}, Test={test_acc}, Loss={loss}")
        train_acc_map[C][sigma] = train_acc
        test_acc_map[C][sigma] = test_acc
        loss_map[C][sigma] = loss

print('epsilon:', epsilon_map)
print('train_acc:', train_acc_map)
print('test_acc:', test_acc_map)
print('loss:', loss_map)


def plot(arr, name, save=False, z_range=None):
    global C_list, sigma_list
    Z = np.zeros([len(C_list), len(sigma_list)])
    for i, C in enumerate(C_list):
        for j, sigma in enumerate(sigma_list):
            Z[i][j] = arr[C][sigma]
    Z = np.array(Z).reshape(len(C_list), len(sigma_list))

    sigma_l, C_l = np.meshgrid(sigma_list, C_list)  # match Z indexing
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(C_l, sigma_l, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('C')
    ax.set_ylabel('sigma')
    ax.set_zlabel(name)
    if z_range:
        ax.set_zlim(0, z_range)
    else:
        ax.set_zlim(np.min(Z), np.max(Z))

    ax.zaxis.set_major_locator(LinearLocator(10))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if save:
        os.makedirs('figs', exist_ok=True)
        plt.savefig(f'figs/{name}.png')
    plt.show()


plot(train_acc_map, 'DP_exp_train_acc', save=True)
plot(test_acc_map, 'DP_exp_test_acc', save=True)
plot(epsilon_map, 'DP_exp_epsilon', save=True)
