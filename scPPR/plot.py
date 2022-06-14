import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot_qt(ax, theta, n_plot=100):
    def cuq(n, x):
        tq = np.zeros((len(n), 3))
        for i, k in enumerate(n):
            tq[i] = np.quantile(x[:k], (0.025, 0.5, 0.975))
        return tq
    n_rep = len(theta)
    n_iter = np.round(np.linspace(100, n_rep, n_plot)).astype(np.int)
    tq = cuq(n_iter, theta)
    ax.plot(range(n_rep), theta, color='lightgrey')
    ax.plot(n_iter, tq[:,2], "--", color='black')
    ax.plot(n_iter, tq[:,1], color='black')
    ax.plot(n_iter, tq[:,0], "--", color='black')

def plot_inference(samples_phase):
    n_biomarkers = samples_phase.shape[0]
    f, axes = plt.subplots(n_biomarkers-1, 2, figsize=(10, 5))
    for i in range(n_biomarkers-1):
        plot_qt(axes[i, 0], samples_phase[i+1,:])
        axes[i, 0].set_title('Eigengene {}'.format(i+2))
        axes[i, 0].set_xlabel('Iteration')
        axes[i, 0].set_ylabel('Phase')
        axes[i, 1].hist(samples_phase[i+1,:], bins=24, range=(0, 2*np.pi))
        axes[i, 1].set_yscale("log")
        axes[i, 1].set_xlabel('Phase')
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 0].spines['top'].set_visible(False)
        axes[i, 0].spines['right'].set_visible(False)
        axes[i, 1].spines['top'].set_visible(False)
        axes[i, 1].spines['right'].set_visible(False)
    plt.tight_layout()

def plot_bayes_score(bayes_scores, individual_phases):
    bayes_scores['Phase'] = individual_phases
    bayes_scores = bayes_scores.sort_values(by="Phase")
    tick_bins = np.linspace(0, 2 * np.pi, 5)
    ticks = ['0', r'$\pi$/2', r'$\pi$', r'3$\pi$/2', r'2$\pi$']
    color_map = {"G1.score": "red", "S.score": "green", "G2M.score": "blue"}
    for stage, color in color_map.items():
        plt.plot(bayes_scores['Phase'], bayes_scores[stage], label=stage.replace('.', ' '), color=color, marker='o', alpha=0.5)
    plt.xlabel('Phase')
    plt.ylabel('Bayes score')
    plt.xticks(tick_bins, ticks, rotation=0)
    plt.tick_params(labelsize=10)
    plt.legend(loc='best')
    plt.show()