import numpy as np


def spiral_data_2d(n_classes=6, n_samples=1000, arm_length=1, noise_scale=0.02):
    angular_lst= np.linspace(0, 2*np.pi / n_classes, n_samples)
    arm_lengths = np.linspace(0.0, arm_length, n_samples)
    output = np.zeros((n_classes, n_samples, 2))
    for i in range(n_classes):
        agls = angular_lst + 2 * i *  np.pi / n_classes
        output[i,:,0] = arm_lengths * np.cos(agls) + np.random.normal(0, noise_scale, n_samples)
        output[i,:,1] = arm_lengths * np.sin(agls) + np.random.normal(0, noise_scale, n_samples)
    return output

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    points = np.reshape(spiral_data_2d(), (-1, 2))
    # points = spiral_data_2d().view(-1, 2)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(points[:,0], points[:,1])
    plt.show()
