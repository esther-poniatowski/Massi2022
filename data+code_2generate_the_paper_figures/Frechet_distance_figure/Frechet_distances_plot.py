import numpy as np
import copy
import matplotlib.pyplot as plt
import parameters_MF_MB as PRM
import simulations_MF_MB as SIM
from figures_indiv import extract_data
import analyzes_MF_MB as ALY
import figures_utils as fig_utl
import matplotlib.patches as mpatches
import similaritymeasures
import matplotlib_latex_bridge as mlb

mlb.setup_page(textwidth=6.97522, columnwidth=3.36305, fontsize=12)

# Methods for replays
convergence = True
method_p = 'predecessor'

H_rpls = {}
df_best = {}
dataset_dt = {}
data_stat = {}
Q_rpls = {}

std = {}
mean_df = {}

labels = ['No replay', 'Backward replay', 'Shuffled replay', 'Prioritized sweeping']

colors_replays = {0: 'royalblue',
                  1: 'orange',
                  2: 'forestgreen',
                  3: 'orchid',
                  4: 'black'}

for det in [True, False]:
    if det:
        env = '_D'
    else:
        env = '_S'

    params = copy.deepcopy(PRM.params)
    params['replay_refs'] = [0,1,2,4]

    Dl0 = SIM.recover_data('Dl0'+env)
    LCl = SIM.recover_data('LCl'+env)
    LCl0, LCl1, _, _ = ALY.split_before_after_change(LCl, params=params)
    SIM.save_data(LCl0, 'LCl0'+env)
    SIM.save_data(LCl1, 'LCl1'+env)

    i_repr = ALY.identify_representative(Dl0, LCl0, params=params)

    Dl_indiv, LC_indiv, Model_indiv = SIM.get_individual_data_per_trial(i_indiv=i_repr, params=params,
                                                                        convergence=convergence, method_p=method_p)

    n_trial_init = 0
    n_trial_end = 24

    if det:
        best = [(35, 4, 28, 0), (28, 6, 26, 0), (26, 5, 1, 0), (1, 6, 23, 0), (23, 7, 10, 0), (10, 7, 22, 1)]
    else:
        best = [(35, 6, 33, 0), (33, 6, 17, 0), (17, 5, 16, 0), (16, 6, 7, 0), (7, 5, 9, 0), (9, 6, 22, 1)]

    H_rpls[env] = []
    Q_rpls[env] = []
    for t in range(n_trial_init, n_trial_end):
        Q_rpls[env].append(Dl_indiv[t]['Q_explo'])
        H_rpls[env].append(Dl_indiv[t]['h_explo'])

    colors = np.random.random((n_trial_end - n_trial_init, 3))

    x_states, y_states, T = extract_data(det, params=params)

    df_best[env] = {}
    for t_r in params['replay_refs']:
        df_best[env][t_r] = []
        for t in range(n_trial_init, n_trial_end):
            traj_det = H_rpls[env][t][t_r]
            df_best[env][t_r].append((similaritymeasures.frechet_dist([(x_states[tr[0]], y_states[tr[0]]) for tr in best],
                                                           [(x_states[tr[0]], y_states[tr[0]]) for tr in traj_det])))

    df = df_best[env]

    median_df = []
    dataset_dt[env] = []
    label_dt = []
    percentile_25 = []
    percentile_75 = []
    colors = []
    std[env] = []
    mean_df[env] = []
    data_stat[env] = {}
    for idd, t_r in enumerate(params['replay_refs']):
        dataset_dt[env].append(df_best[env][t_r])
        label_dt.append(labels[idd])
        median_df.append(np.median(df[t_r]))
        percentile_25.append(np.percentile(df[t_r], [25, 50, 75])[0])
        percentile_75.append((np.percentile(df[t_r], [25, 50, 75])[2]))
        data_stat[env][t_r] = (df_best[env][t_r], np.std(df[t_r]))

        colors.append(colors_replays[t_r])
        std[env].append(np.std(df[t_r]))
        mean_df[env].append(np.mean(df[t_r]))

params = copy.deepcopy(PRM.params)
params['replay_refs'] = [0, 1, 2, 4]

labels = ['No replay', 'Backward replay', 'Shuffled replay', 'Prioritized sweeping']

fig = mlb.figure_textwidth(widthp=1, height=3)
axs = fig.subplots(1, 2)


for id_env, envi in enumerate(['_D', '_S']):
    ax = axs[id_env]

    parts = ax.violinplot(dataset_dt[envi], positions=range(len(df_best[envi])), showmeans=False, showmedians=False,
                          showextrema=False)

    for idp, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[idp])
        pc.set_label(labels[idp])
        pc.set_edgecolor('white')
        pc.set_alpha(0.5)

    ax.bar(range(len(df_best[envi])), mean_df[envi], yerr=std[envi], alpha=0, align='center', ecolor='black',
           capsize=5)
    ax.set_xlabel('Replay types')

    ax.scatter(range(len(df_best[envi])), mean_df[envi], color='white', edgecolor='k', zorder=100)
    fig_utl.hide_spines(ax, sides=['right', 'top', 'bottom'])
    fig_utl.hide_ticks(ax, 'x')

    if id_env == 0:
        ax.set_ylabel('Fréchet distances to\nthe optimal trajectory (m)')

    if id_env == 1:
        fig_utl.hide_all(ax)
        fig_utl.hide_ticks(ax, 'y')

    ax.set_ylim(0, 1.6)

handles = [mpatches.Patch(facecolor=c, label=l) for l, c in zip(labels, colors)]
legend = ax.legend(handles=handles, loc=2, bbox_to_anchor=(1, 1))
plt.savefig('frechet_distance_figure.pdf')

