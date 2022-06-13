import copy
import parameters_MF_MB as PRM
import figures_final_alpha as GRAPH
import matplotlib_latex_bridge as mlb


mlb.setup_page(textwidth=6.97522, columnwidth=3.36305, fontsize=10)

# # ALPHA ANALYSIS FIGURE ##########################
params = copy.deepcopy(PRM.params)
params['replay_refs'] = [0,1,2,4]
GRAPH.figure_alpha_selection(params=params)

