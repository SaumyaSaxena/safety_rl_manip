import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

if __name__ == "__main__":

    k_variance_success = False
    k_variance_collision = False
    irrelevant_collision = False
    full_state = True

    if k_variance_success:
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
        colors = {3: (0, 72/255, 166/255), 4: 'green', 5: 'orange', 6: 'purple'}

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

    if k_variance_collision:
        k_values = [2, 3, 4, 5, 6]

        collision_rate_static = {
            3: [22.1, 0],
            4: [49.8, 22.6, 0],
            5: [60.7, 41.8, 17.3, 0],
            6: [67.6, 47.1, 32.3, 11.3, 0],
        }
        collision_rate_dynamic = {
            3: [0.23, 0],
            4: [0.64, 0.0, 0],
            5: [1.9, 0.05, 0.0, 0],
            6: [2.6, 0.0, 0.05, 0.0, 0],
        }

        # Colors for different values of N
        N = [3,4,5,6]
        colors = {3: 'blue', 4: 'green', 5: 'orange', 6: 'purple'}

        fontsize = 15
        linewidth = 3
        # Plotting
        plt.figure(figsize=(10, 6))
        for n, rates in collision_rate_static.items():
            plt.plot(k_values[:len(rates)], rates, linestyle=':', color=colors[n], label=f'Static kNN (N={n})', linewidth=linewidth)
            plt.scatter(k_values[:len(rates)], rates, color=colors[n], zorder=5, facecolors='none')
            # Annotate the value of N
            text_loc = [-0.1, -2.5] if n==6 else [0.04, 0.0]
            plt.text(k_values[len(rates)-1]+text_loc[0], rates[-1]+text_loc[1], f'N={n}', color=colors[n], fontsize=fontsize, ha='left', va='center')

        for n, rates in collision_rate_dynamic.items():
            plt.plot(k_values[:len(rates)], rates, linestyle='-', color=colors[n], label=f'Dynamic kNN (N={n})', linewidth=linewidth)
            plt.scatter(k_values[:len(rates)], rates, color=colors[n], zorder=5)
            # Annotate the value of N
            text_loc = [-0.1, -2.5] if n==6 else [0.04, 0.0]
            plt.text(k_values[len(rates)-1]+text_loc[0], rates[-1]+text_loc[1], f'N={n}', color=colors[n], fontsize=fontsize, ha='left', va='center')

        # Add labels, title, and legend
        plt.xlabel(r'$N_{\text{rel}}$', fontsize=fontsize)
        plt.ylabel('Collision Rate w/ unselected objects (%)', fontsize=fontsize)
        # plt.title('Success Rate vs k for Static and Dynamic kNN', fontsize=14)
        plt.xticks(ticks=k_values, labels=k_values)
        # plt.yticks(ticks=np.linspace(20,100,5), labels=np.linspace(20,100,5))
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
        ax.legend(handles=legend_elements2, fontsize=fontsize, loc='upper right', bbox_to_anchor=(1, 0.85))

        # Show the plot
        plt.tight_layout()
        path = "/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/results"
        plt.savefig(f"{path}/irrelevant_collision_rate_vs_k.png", dpi=300)
        plt.close()

    if full_state:
        N = [1,2,3,4,5,6]
        lagrange_succ = [63.73, 56.63, 47.39, 30.21, 26.49, 16.89]
        RARL_succ = [99.55, 95.55, 81.80, 75.43, 53.41, 44.93]

        lagrange_fail = [10.03, 16.32, 10.52, 25.68, 42.18, 48.84]
        RARL_fail = [0.45, 0.25, 1.75, 2.65, 0.95, 22.84]

        fontsize = 15
        linewidth = 3
        colors = {'lagrange': (0, 72/255, 166/255), 'rarl': 'green'}

        plt.figure(figsize=(10, 6))
        plt.plot(N, lagrange_succ, linestyle='-', color=colors['lagrange'], label=f'Vanilla-RL', linewidth=linewidth)
        plt.scatter(N, lagrange_succ, color=colors['lagrange'], zorder=5)
        plt.plot(N, RARL_succ, linestyle='-', color=colors['rarl'], label=f'RARL', linewidth=linewidth)
        plt.scatter(N, RARL_succ, color=colors['rarl'], zorder=5)

        plt.plot(N, lagrange_fail, linestyle=':', color=colors['lagrange'], label=f'Vanilla-RL', linewidth=linewidth)
        plt.scatter(N, lagrange_fail, color=colors['lagrange'], zorder=5, facecolors='none')
        plt.plot(N, RARL_fail, linestyle=':', color=colors['rarl'], label=f'RARL', linewidth=linewidth)
        plt.scatter(N, RARL_fail, color=colors['rarl'], zorder=5, facecolors='none')

        # Add labels, title, and legend
        plt.xlabel(r'$N$', fontsize=fontsize)
        plt.ylabel('Success Rate (%)', fontsize=fontsize)
        plt.xticks(ticks=N, labels=N)
        # plt.yticks(ticks=np.linspace(20,100,5), labels=np.linspace(20,100,5))
        plt.tick_params(axis='x', labelsize=fontsize)
        plt.tick_params(axis='y', labelsize=fontsize)

        legend_elements1 = [
            Line2D([0], [0], linestyle='-', color=colors['lagrange'], label='Vanilla-RL', linewidth=linewidth),
            Line2D([0], [0], linestyle='-', color=colors['rarl'], label='RARL', linewidth=linewidth),
        ]

        ax = plt.gca()
        legend1 = ax.legend(handles=legend_elements1, fontsize=15, loc='upper right', bbox_to_anchor=(1, 1))
        ax.add_artist(legend1)

        legend_elements2 = [
            Line2D([0], [0], linestyle='-', color='black', label='Success rate (%)', linewidth=linewidth),
            Line2D([0], [0], linestyle=':', color='black', label='Top box dynamic \n int. failure (%)', linewidth=linewidth),
        ]
        ax.legend(handles=legend_elements2, fontsize=15, loc='upper right', bbox_to_anchor=(1, 0.85))

        # Show the plot
        plt.tight_layout()
        path = "/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/results"
        plt.savefig(f"{path}/lagrange_rarl.png", dpi=300)
        plt.close()

    if irrelevant_collision:
        fontsize = 15
        k_values = [2, 3, 4, 5]

        # Data for Static-kNN and Dynamic-kNN (each with 4 stacked components)
        static_knn_data = [
            [29.9, 0.2, 67.57, 2.40],  # for k=2
            [37.6, 0.8, 47.10, 14.51],  # for k=3
            [51.91, 2.26, 32.26, 13.57],  # for k=4
            [50.3, 0.50, 11.37, 37.42],  # for k=5
        ]

        dynamic_knn_data = [
            [45.2, 0.2, 2.60, 52.05],  # for k=2
            [40.6, 0.8, 0.00, 58.70],  # for k=3
            [59.18, 1.86, 0.05, 38.91],  # for k=4
            [50.7, 0.45, 0.00, 49.27],  # for k=5
        ]

        # Convert to NumPy arrays for easier stacking
        static_knn_data = np.array(static_knn_data).T  # Transpose to stack vertically
        dynamic_knn_data = np.array(dynamic_knn_data).T

        # Define bar width and x positions
        bar_width = 0.35
        bar_dist = 0.03
        x = np.arange(len(k_values))  # Positions for bars

        # Define colors for each stacked component
        colors = [(0, 72/255, 166/255), (255/255, 149/255, 0), (205/255, 179/255, 255/255), (133/255, 15/255, 103/255)]
        labels = ['Success', 'Top box OOB', 'Collision (unselected)', 'Collision (selected)']

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot Static-kNN stacked bars
        bottom_static = np.zeros(len(k_values))
        bottom_dynamic = np.zeros(len(k_values))

        # import ipdb; ipdb.set_trace()
        for i in range(len(labels)):  # Loop over components
            bars_static = ax.bar(x-bar_width/2-bar_dist, static_knn_data[i], width=bar_width, color=colors[i], bottom=bottom_static, label=labels[i], edgecolor='black')
            bars_dynamic = ax.bar(x+bar_width/2+bar_dist, dynamic_knn_data[i], width=bar_width, color=colors[i], bottom=bottom_dynamic, edgecolor='black')
            bottom_static += static_knn_data[i]
            bottom_dynamic += dynamic_knn_data[i]

            # Add percentage value annotations
            for bar in bars_static:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2, f'{height:.1f}', ha='center', va='center', fontsize=10, color='white')
            for bar in bars_dynamic:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2, f'{height:.1f}', ha='center', va='center', fontsize=10, color='white')

        x_ticks = np.concatenate([x - bar_width/2-bar_dist, x + bar_width/2+bar_dist])  # Positions for Static and Dynamic
        x_labels = ['Static\nkNN'] * len(k_values) + ['Dynamic\nkNN'] * len(k_values)  # Labels for each bar

        # Labels, title, and legend
        ax.set_xlabel(r'$N_{\text{rel}}$', fontsize=fontsize)
        ax.set_ylabel('Percentage (%)', fontsize=fontsize)

        # Set x-axis ticks for Static-kNN and Dynamic-kNN
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=10)

        # Set minor ticks for k values
        ax.set_xticks(x, minor=True)
        ax.set_xticklabels(k_values, minor=True, fontsize=fontsize)

        # Adjust tick positions
        ax.tick_params(axis='x', which='major', pad=0)  # Move Static/Dynamic labels slightly up
        ax.tick_params(axis='x', which='minor', pad=20)  # Move k labels even lower
        ax.tick_params(axis='y', labelsize=fontsize)

        ax.legend(loc='upper center', ncols=4, bbox_to_anchor=(0.5, 1.016))
        # ax.set_title('Stacked Bar Chart for Static-kNN and Dynamic-kNN')

        # Show plot
        plt.tight_layout()
        path = "/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/results"
        plt.savefig(f"{path}/irrlevant_stacked1.png", dpi=300)
        plt.close()
