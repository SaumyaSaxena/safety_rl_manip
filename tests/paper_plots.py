import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

if __name__ == "__main__":

    k_values = [2, 3, 4, 5, 6]
    success_rate_static = {
        3: [75.3, 81.8],
        4: [46.16, 57.8, 75.4],
        5: [37.3, 40.6, 64.7, 53.4],
        6: [29.9, 37.6, 51.9, 50.3, 44.9],
    }
    success_rate_dynamic = {
        3: [84.6, 81.8],
        4: [60, 60.3, 75.4],
        5: [52.6, 45.5, 69.3, 53.4],
        6: [45.2, 40.6, 59.8, 50.7, 44.9],
    }

    # Colors for different values of N
    N = [3,4,5,6]
    colors = {3: 'blue', 4: 'green', 5: 'orange', 6: 'purple'}

    fontsize = 15
    linewidth = 3
    # Plotting
    plt.figure(figsize=(10, 6))
    for n, rates in success_rate_static.items():
        plt.plot(k_values[:len(rates)], rates, linestyle=':', color=colors[n], label=f'Static kNN (N={n})', linewidth=linewidth)
        plt.scatter(k_values[:len(rates)], rates, color=colors[n], zorder=5, facecolors='none')
        # Annotate the value of N
        text_loc = [-0.1, -2.5] if n==6 else [0.04, 0.0]
        plt.text(k_values[len(rates)-1]+text_loc[0], rates[-1]+text_loc[1], f'N={n}', color=colors[n], fontsize=fontsize, ha='left', va='center')

    for n, rates in success_rate_dynamic.items():
        plt.plot(k_values[:len(rates)], rates, linestyle='-', color=colors[n], label=f'Dynamic kNN (N={n})', linewidth=linewidth)
        plt.scatter(k_values[:len(rates)], rates, color=colors[n], zorder=5)
        # Annotate the value of N
        text_loc = [-0.1, -2.5] if n==6 else [0.04, 0.0]
        plt.text(k_values[len(rates)-1]+text_loc[0], rates[-1]+text_loc[1], f'N={n}', color=colors[n], fontsize=fontsize, ha='left', va='center')

    # Add labels, title, and legend
    plt.xlabel(r'$N_{\text{rel}}$', fontsize=fontsize)
    plt.ylabel('Success Rate (%)', fontsize=fontsize)
    # plt.title('Success Rate vs k for Static and Dynamic kNN', fontsize=14)
    plt.xticks(ticks=k_values, labels=k_values)
    plt.yticks(ticks=np.linspace(20,100,5), labels=np.linspace(20,100,5))
    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
    # plt.grid(True, linestyle='--', alpha=0.7)

    legend_elements = [
        Line2D([0], [0], linestyle=':', color='black', label='Static kNN', linewidth=linewidth),
        Line2D([0], [0], linestyle='-', color='black', label='Dynamic kNN', linewidth=linewidth),
    ]
    ax = plt.gca()
    legend1 = ax.legend(handles=legend_elements, fontsize=15, loc='upper right', bbox_to_anchor=(1, 1))
    ax.add_artist(legend1)

    legend_elements2 = []
    for n in N:
        legend_elements2.append(Line2D([0], [0], linestyle='-', color=colors[n], label=f'N={n}', linewidth=linewidth))
    ax.legend(handles=legend_elements2, fontsize=15, loc='upper right', bbox_to_anchor=(1, 0.85))

    # Show the plot
    plt.tight_layout()
    path = "/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/results"
    plt.savefig(f"{path}/success_rate_vs_k.png", dpi=300)
    plt.close()
