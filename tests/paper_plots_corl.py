import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import matplotlib as mpl

if __name__ == "__main__":

    # k_variance_success = False
    # k_variance_collision = False
    # irrelevant_collision = False
    # mpl.rcParams['font.family'] = 'Times'

    mpl.rcParams['font.family'] = 'serif' 
    
    obj_gen = False
    obj_gen_three_plots = True
    const_gen = True

    if obj_gen:
        N = [2, 4, 6, 8, 10, 12]
        noMask = [85.5, 70, 48.5, 25]
        mask = [95.5, 90.5, 79, 66.5]

        noMask_fail = [3.5, 9, 17.5, 20]
        mask_fail = [0.5, 2, 0, 1.5]

        noMask_fail_coll = [8.5, 17, 32.5, 52]
        mask_fail_coll = [4, 5, 18.5, 30.5]

        fontsize = 15
        linewidth = 3
        # colors = {'noMask': (0, 72/255, 166/255), 'mask': 'green'}
        colors = {
            'mask': '#ff9500',       # orange
            'noMask': '#850f67'   # dark magenta
        }

        train_no_mask = 85.5
        train_mask = 95.5

        plt.figure(figsize=(10, 6))
        plt.plot(N, noMask, linestyle='-', color=colors['noMask'], label=f'VL-SafeT-NoMask', linewidth=linewidth)
        plt.scatter(N, noMask, color=colors['noMask'], zorder=5)
        plt.plot(N, mask, linestyle='-', color=colors['mask'], label=f'VL-SafeT', linewidth=linewidth)
        plt.scatter(N, mask, color=colors['mask'], zorder=5)

        plt.plot(N, noMask_fail, linestyle=':', color=colors['noMask'], label=f'VL-SafeT-NoMask', linewidth=linewidth)
        plt.scatter(N, noMask_fail, color=colors['noMask'], zorder=5, facecolors='none')
        plt.plot(N, mask_fail, linestyle=':', color=colors['mask'], label=f'VL-SafeT', linewidth=linewidth)
        plt.scatter(N, mask_fail, color=colors['mask'], zorder=5, facecolors='none')

        plt.plot(N, noMask_fail_coll, linestyle='-.', color=colors['noMask'], label=f'VL-SafeT-NoMask', linewidth=linewidth)
        plt.scatter(N, noMask_fail_coll, color=colors['noMask'], zorder=5, facecolors='none')
        plt.plot(N, mask_fail_coll, linestyle='-.', color=colors['mask'], label=f'VL-SafeT', linewidth=linewidth)
        plt.scatter(N, mask_fail_coll, color=colors['mask'], zorder=5, facecolors='none')

        plt.scatter(6, train_no_mask, color=colors['noMask'], marker='*', s=150, zorder=6)
        plt.scatter(6, train_mask, color=colors['mask'], marker='*', s=150, zorder=6)

        # Add labels, title, and legend
        plt.xlabel('Number of objects', fontsize=fontsize)
        plt.ylabel('Percentage (%)', fontsize=fontsize)
        plt.xticks(ticks=N, labels=N)
        # plt.yticks(ticks=np.linspace(20,100,5), labels=np.linspace(20,100,5))
        plt.tick_params(axis='x', labelsize=fontsize)
        plt.tick_params(axis='y', labelsize=fontsize)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        legend_elements1 = [
            Line2D([0], [0], linestyle='-', color=colors['noMask'], label='VL-TSafe-NoMask', linewidth=linewidth),
            Line2D([0], [0], linestyle='-', color=colors['mask'], label='VL-TSafe', linewidth=linewidth),
        ]

        legend1 = ax.legend(handles=legend_elements1, fontsize=15, loc='upper right', bbox_to_anchor=(1, 1))
        ax.add_artist(legend1)

        legend_elements2 = [
            Line2D([0], [0], linestyle='-', color='black', label=r'$s_{\text{safe}}$', linewidth=linewidth),
            Line2D([0], [0], linestyle=':', color='black', label=r'$f_{\text{dyn}}$', linewidth=linewidth),
            Line2D([0], [0], linestyle='-.', color='black', label=r'$f_{\text{coll}}$', linewidth=linewidth),
        ]
        ax.legend(handles=legend_elements2, fontsize=15, loc='upper right', bbox_to_anchor=(1, 0.85))

        # Show the plot
        plt.tight_layout()
        path = "/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/results"
        plt.savefig(f"{path}/gen_obj_corl.png", dpi=300)
        plt.close()

    if obj_gen_three_plots:
        N = [2, 4, 6, 8, 10, 12]
        noMask = [97.5, 94.5, 85.5, 70, 48.5, 25]
        mask = [98, 97, 95.5, 90.5, 79, 66.5]

        noMask_fail = [0.5, 2, 3.5, 9, 17.5, 20]
        mask_fail = [1, 1, 0.5, 2, 0, 1.5]

        noMask_fail_coll = [1, 3, 8.5, 17, 32.5, 52]
        mask_fail_coll = [1, 2, 4, 5, 18.5, 30.5]

        train_idx = 2

        fontsize = 17
        linewidth = 3
        # colors = {'noMask': (0, 72/255, 166/255), 'mask': 'green'}
        colors = {
            'mask': '#ff9500',       # orange
            'noMask': '#850f67'   # dark magenta
        }

        # titles = [
        #     r'$SafeSucc \; \% (\uparrow) $',
        #     r'$StackFail \; \% (\downarrow)$',
        #     r'$ObjFail \; \% (\downarrow)$'
        # ]
        titles = [
            r'SafeSucc $ \% \; (\uparrow) $',
            r'StackFail $ \% \; (\downarrow)$',
            r'ObjFail $ \% \; (\downarrow)$'
        ]

        data_pairs = [
            (noMask, mask),
            (noMask_fail, mask_fail),
            (noMask_fail_coll, mask_fail_coll)
        ]

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        for i, ax in enumerate(axes):
            nm_data, m_data = data_pairs[i]

            ax.plot(N, nm_data, linestyle='-', color=colors['noMask'], linewidth=linewidth)
            ax.scatter(N, nm_data, color=colors['noMask'], zorder=5)

            ax.plot(N, m_data, linestyle='-', color=colors['mask'], linewidth=linewidth)
            ax.scatter(N, m_data, color=colors['mask'], zorder=5)

            ax.set_title(titles[i], fontsize=fontsize)
            ax.set_xlabel('Test # objects', fontsize=fontsize)
            ax.set_xticks(N)
            ax.tick_params(axis='both', labelsize=fontsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.scatter(N[train_idx], nm_data[train_idx], color=colors['noMask'], marker='*', s=500, zorder=6)
            ax.scatter(N[train_idx], m_data[train_idx], color=colors['mask'], marker='*', s=500, zorder=6)
            ax.axvline(x=N[train_idx], linestyle=':', color='black', linewidth=2)

        axes[0].set_ylabel('Percentage (%)', fontsize=fontsize)

        # Add legends only once
        legend_elements_model = [
            Line2D([0], [0], linestyle='-', color=colors['noMask'], label='VLTSafe-NoMask', linewidth=linewidth),
            Line2D([0], [0], linestyle='-', color=colors['mask'], label='VLTSafe', linewidth=linewidth),
            Line2D([0], [0], marker='*', color='black', label='Train', markersize=15, linestyle='None')
        ]
        axes[2].legend(handles=legend_elements_model, fontsize=fontsize, loc='upper right')

        plt.tight_layout()
        path = "/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/results"
        plt.savefig(f"{path}/gen_obj_corl_split.png", dpi=300)
        plt.close()


    if const_gen:
        N = [6, 8, 10, 12]
        preset = [67, 59.5, 53, 42.5]
        randomConst = [78.5, 67, 56, 45]

        train_preset = 79.5
        train_randomConst = 78.5

        fontsize = 24
        linewidth = 3
        # colors = {'preset': (0, 72/255, 166/255), 'randomConst': 'green'}

        colors = {
            'randomConst': '#0d948f',       # orange
            'preset': '#740cad'   # dark magenta
        }

        plt.figure(figsize=(10, 6))
        plt.plot(N, preset, linestyle='-', color=colors['preset'], label=r'$\pi_{\text{FixedConst (N=6)}}$', linewidth=linewidth)
        plt.scatter(N, preset, color=colors['preset'], zorder=5)
        plt.plot(N, randomConst, linestyle='-', color=colors['randomConst'], label=r'$\pi_{\text{RandomConst (N=6)}}$', linewidth=linewidth)
        plt.scatter(N, randomConst, color=colors['randomConst'], zorder=5)

        plt.scatter(6, train_preset, color=colors['preset'], marker='*', s=500, zorder=6)
        plt.scatter(6, train_randomConst, color=colors['randomConst'], marker='*', s=500, zorder=6)

        # Add labels, title, and legend
        plt.xlabel('Test # objects', fontsize=fontsize)
        plt.ylabel('SafeSucc %', fontsize=fontsize)
        plt.xticks(ticks=N, labels=N)
        # plt.yticks(ticks=np.linspace(20,100,5), labels=np.linspace(20,100,5))
        plt.tick_params(axis='x', labelsize=fontsize)
        plt.tick_params(axis='y', labelsize=fontsize)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title('Test performance in RandomConst domain', fontsize=fontsize)

        legend_elements1 = [
            Line2D([0], [0], linestyle='-', color=colors['preset'], label='FixedConst', linewidth=linewidth),
            Line2D([0], [0], linestyle='-', color=colors['randomConst'], label='RandomConst', linewidth=linewidth),
            Line2D([0], [0], marker='*', color='black', label='Train', markersize=22, linestyle='None'),
        ]

        legend1 = ax.legend(handles=legend_elements1, fontsize=fontsize, loc='upper right', bbox_to_anchor=(1, 1))
        ax.add_artist(legend1)

        # Show the plot
        plt.tight_layout()
        path = "/home/saumyas/Projects/safe_control/safety_rl_manip/outputs/media/results"
        plt.savefig(f"{path}/gen_const_comp.png", dpi=300)
        plt.close()