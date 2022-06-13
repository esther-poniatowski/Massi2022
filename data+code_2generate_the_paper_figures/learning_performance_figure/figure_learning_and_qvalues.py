import copy
import matplotlib.pyplot as plt
import parameters_MF_MB as PRM
import figures_final as GRAPH
import matplotlib_latex_bridge as mlb

mlb.setup_page(textwidth=6.97522, columnwidth=3.36305, fontsize=10)


def main():

	##################################################
	# LEARNING PLOT AND VIOLIN STATISTICAL ANALYSIS FIGURE ##########################
	params = copy.deepcopy(PRM.params)
	params['replay_refs'] = [0,1,2,4]
	fig_det = GRAPH.figure_learning_curves_violin_plots(det=True, params=params, thres=0.05)
	fig_nondet = GRAPH.figure_learning_curves_violin_plots(det=False, params=params, thres=0.05)
	fig_det.savefig("Saved_figures/learning_plots_det_1200.jpg", format='jpg', dpi=1200)
	fig_nondet.savefig("Saved_figures/learning_plots_nodet_1200.jpg", format='jpg', dpi=1200)
	# plt.show()

	##################################################

	# # Q-VALUES AND REPLAYS ANALYSIS FIGURE ##########################
	params = copy.deepcopy(PRM.params)
	params['replay_refs'] = [0,1,2,4]
	fig_det = GRAPH.figure_Qvalues(det=True, params=params, legends=False)
	fig_nondet = GRAPH.figure_Qvalues(det=False, params=params, legends=True)
	fig_det.savefig("Saved_figures/qvalues_det.pdf")
	fig_nondet.savefig("Saved_figures/qvalues_nodet.jpg", format='jpg', dpi=300)
	# plt.show()

if __name__ == '__main__':
	main()

