import matplotlib.pyplot as plt
import numpy as np


def create_bar_plot():
    # Given values for different parameter configurations
    losses = {'CL': [0.9757, 0.9652, 0.9685], 'NCA': [0.9673, 0.9494, 0.9511], 'CF': [0.9587, 0.9448, 0.9500]}
    distances = {'L2': [0.8418, 0.9417, 0.9444], 'SNR': [0.8880, 0.9436, 0.9396], 'L1': [0.7607, 0.9304, 0.9172]}
    margins = {'m1': [0.2473, 0.3925, 0.3744], 'm06': [0.8798, 0.9441, 0.9405], 'm03': [0.8403, 0.9368, 0.9399],
               'm0': [0.8418, 0.9417, 0.9444]}

    l2_m0_map = [losses[key][0] for key in losses]
    l2_m0_p1 = [losses[key][1] for key in losses]
    l2_m0_p5 = [losses[key][2] for key in losses]

    # Define x-axis positions for the groups
    x = np.arange(len(losses))

    # Define width of the bars
    bar_width = 0.1

    # Define color palette
    colors = ['#3D5A80', '#98D1D9', '#EE6C4D', '#E0FBFC']

    legend_labels = ['CL', 'NCA', 'CF']
    xticks = ['mAP', 'P@1', 'P@5', '']

    for i, (label, values) in enumerate(losses.items()):
        plt.bar(x/1.5 + i * bar_width, values, width=bar_width, label=label, color=colors[i])

    plt.xlim(x[0] - bar_width, x[-1] + len(losses) * bar_width)
    group_center_positions = x/1.5 + 0.5 * bar_width * (len(losses) - 1)
    plt.xticks(group_center_positions, xticks)


    # Set y-axis limits
    plt.ylim(0.8, 1)

    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Losses for distance L2 & margin = 0')
    plt.legend(legend_labels)
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('/export/home/group02/C5-G2/Week3/bar_plots/Losses.png')


create_bar_plot()
