#!/usr/bin/env python

import numpy as np
import os.path
from os import path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.spatial as scispa
import math
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib_latex_bridge as mlb

mlb.setup_page(textwidth=6.97522, columnwidth=3.36305, fontsize=11)
mlb.figure_textwidth(widthp=0.5)


def create_voronoid(transition_path="transitions.txt"):

	p=60 
	states_=[(0.00320002,0.0059351),(0.310727,0.0241474), (0.593997,0.152759),(0.724483, -0.118826),(0.956215, 0.0761724),(-0.0546479, 0.308712),(-0.359653, 0.3537),(-0.465333, 0.651306),(-0.636187 ,0.920479),(-0.325808 ,0.96138),(-0.0262775, 0.936864),(-0.12853 ,0.608485),(0.275655, 0.947168),(0.164439 ,1.22664),(-0.588397, 1.21822),(-0.779392, 0.652651),(-0.669064, 0.253532),(-0.863761 ,0.0191329),(-0.924563, -0.277856),(-1.08771, 0.22059),(-1.23553, 0.490936),(-1.09893 ,0.758705),(-0.293732, 1.30145),(0.208853, 0.454286),(0.522142, 0.755471),(0.69987, 0.513679),(-0.0588719, -0.289781),( 0.0889117 ,-0.583929),(-0.120961, -0.804133),(-0.939397, 1.014),( 0.367168, -0.274708),( -0.329299 ,-0.156936),(0.39444, -0.660872),(-0.539525, -0.381784),(-1.22956, -0.263396),(-0.504464 ,-0.834182)] #position of the centers of the states

	x_states=[]
	y_states=[]
	for x, y in states_ : 
		x_states.append(x)
		y_states.append(y)


	############ load the transition matrix by using the transition file ############
	'''
	The transition matrix M has 3 dimensions, with shape (number_of_states,number_of_actions,number_of_states), such as M[state0,a,state1] contains the probability that the robot ended in state1 when performing action a in state0.
	'''
	M = np.zeros((p, 8, p))
	with open(transition_path) as transitions_file:
		for k, line in enumerate(transitions_file):
			line = line.strip()
			l_split = line.split(",")
			if len(l_split) < 4:
				continue
			s0 = int(l_split[0])
			a = int(l_split[1])
			s1 = int(l_split[2])
			m = float(l_split[3])
			M[s0, a, s1] = m

	M=M[0:36,:,0:36] #restricitng the matrix to the right states
	#we have a total of 5501 transitions
	print(f"Transitions: {math.prod(M.shape)}")

	for s in range (36): #transforming the number of transitions into probabilities
		for a in range (8) : 
			somme=np.sum(M[s,a,:])
			if somme!=0 :
				M[s,a,:]=M[s,a,:]/somme

	############ entropy map ############
	def entropy (list_proba,b) :
		n=len(list_proba)
		ent=0
		for k in range (n) :
			if list_proba[k]!=0 : 
				ent-=list_proba[k]*np.log(list_proba[k])/np.log(b)
		return(ent)

	entropies=np.zeros((36,8))
	for s in range (36) : 
		for a in range (8) : 
			entropies[s,a]=entropy(M[s,a,:],2)

	max_entropies = np.max(entropies, axis=1)
	print(f"Max entro {max_entropies.shape}")

	centre_states = np.array(states_)
	# add 4 distant dummy points
	centre_states = np.append(centre_states, [[999,999], [-999,999], [999,-999], [-999,-999]], axis = 0)
	vor = scispa.Voronoi(centre_states)

	def NormalizeData(data):
		return (data - np.min(data)) / (np.max(data) - np.min(data))

	norm_max_entr = NormalizeData(max_entropies)
	
	scispa.voronoi_plot_2d(vor, show_vertices=False, show_points=False)
	for r in range(len(vor.point_region)):
		region = vor.regions[vor.point_region[r]]
		if not -1 in region:
			polygon = [vor.vertices[i] for i in region]
			c = plt.fill(*zip(*polygon), color='white', zorder=0)


	# states and legend
	plot_states = np.array(states_)
	plt.scatter(plot_states[35, 0], plot_states[35, 1], c="green", label="35 - initial state")
	plt.scatter(plot_states[22, 0], plot_states[22, 1], c="purple", label="22 - first reward state")
	plt.scatter(plot_states[4, 0], plot_states[4, 1], c="orange", label="4 - second reward state")

	index_state = []
	for i in range(0, 36):
		if i != 4 and i != 22 and i!= 35: 
			index_state.append(i)

	# setting adhoc offsets for each state to not have states' labels over the states' borders
	offset_state_label = np.zeros((36, 2))
	# 14
	offset_state_label[13, 0] = -0.1
	offset_state_label[28, 0] = -0.1
	# 30
	offset_state_label[28, 1] = -0.1
	# 11
	offset_state_label[10, 0] = -0.1
	# 3
	offset_state_label[3, 1] = -0.1
	# 5
	offset_state_label[4, 1] = -0.05
	# 6
	offset_state_label[5, 1] = -0.05
	# 7
	offset_state_label[6, 1] = -0.05
	# 8
	offset_state_label[7, 1] = -0.05
	# 9
	offset_state_label[8, 1] = -0.05
	# 10
	offset_state_label[9, 0] = -0.05
	offset_state_label[9, 1] = -0.05
	# 11
	offset_state_label[10, 1] = -0.05
	# 15
	offset_state_label[14, 1] = -0.05
	offset_state_label[14, 0] = -0.05
	# 16
	offset_state_label[15, 1] = -0.05
	offset_state_label[15, 0] = -0.05
	# 17
	offset_state_label[16, 1] = -0.05
	offset_state_label[16, 0] = -0.05
	# 18
	offset_state_label[17, 1] = -0.1
	offset_state_label[17, 0] = -0.05
	# 19
	offset_state_label[18, 1] = -0.05
	offset_state_label[18, 0] = -0.05
	# 20
	offset_state_label[19, 1] = -0.05
	offset_state_label[19, 0] = -0.05
	# 21
	offset_state_label[20, 1] = -0.05
	offset_state_label[20, 0] = -0.05

	# 23
	offset_state_label[21, 1] = -0.05
	offset_state_label[21, 0] = -0.05
	# 25
	offset_state_label[23, 1] = -0.05
	# 26
	offset_state_label[24, 1] = -0.05
	offset_state_label[24, 0] = -0.05
	# 27
	offset_state_label[25, 1] = -0.05
	offset_state_label[25, 0] = -0.05
	# 28
	offset_state_label[26, 1] = -0.05
	offset_state_label[26, 0] = -0.1
	# 29
	offset_state_label[27, 0] = -0.1
	# 31
	offset_state_label[29, 0] = -0.1
	# 32
	offset_state_label[30, 1] = -0.05
	# 33
	offset_state_label[31, 1] = -0.05
	offset_state_label[31, 0] = -0.1
	# 34
	offset_state_label[32, 0] = -0.1

	for id_el, el in enumerate(index_state):
		plt.text(x=plot_states[el, 0] + offset_state_label[id_el, 0], y= plot_states[el, 1] + offset_state_label[id_el, 1], s=el)

	# plt.legend(facecolor="lightpink")
	plt.legend(loc=(-0.1, 0.9))


	# colorbar
	# plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.hot))


def mask_outside_polygon(poly_verts, ax=None):
	"""
	Plots a mask on the specified axis ("ax", defaults to plt.gca()) such that
	all areas outside of the polygon specified by "poly_verts" are masked.  

	"poly_verts" must be a list of tuples of the verticies in the polygon in
	counter-clockwise order.

	Returns the matplotlib.patches.PathPatch instance plotted on the figure.
	"""
	if ax is None:
		ax = plt.gca()

	# Get current plot limits
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	# Verticies of the plot boundaries in clockwise order
	bound_verts = [(xlim[0], ylim[0]), (xlim[0], ylim[1]), 
				   (xlim[1], ylim[1]), (xlim[1], ylim[0]), 
				   (xlim[0], ylim[0])]

	# A series of codes (1 and 2) to tell matplotlib whether to draw a line or 
	# move the "pen" (So that there's no connecting line)
	bound_codes = [mpath.Path.MOVETO] + (len(bound_verts) - 1) * [mpath.Path.LINETO]
	poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 1) * [mpath.Path.LINETO]

	# Plot the masking patch
	path = mpath.Path(bound_verts + poly_verts, bound_codes + poly_codes)
	patch = mpatches.PathPatch(path, facecolor='white', edgecolor='none', zorder=2)
	patch = ax.add_patch(patch)

	# Reset the plot limits to their original extents
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	return patch


def create_map(map_path="map1.pgm", scale=1.0, offset=np.zeros(2)):
	img=mpimg.imread(map_path)

	# get list of coordinates for the border
	coords = []
	h, w = img.shape
	for i in range(h):
		for j in range(w):
			if img[i,j] == 0:
				coords.append((i*scale, j*scale))
	coords = np.array(coords)

	# center data and adjust offset
	coords[:, 0] -= np.mean(coords[:, 0])
	coords[:, 1] -= np.mean(coords[:, 1])
	coords = coords + offset

	# compute the contourn
	hull = ConvexHull(coords)
	hull_points_x = [coords[v, 0] for v in hull.vertices]
	hull_points_x.append(coords[hull.vertices[0], 0])
	hull_points_y = [coords[v, 1] for v in hull.vertices]
	hull_points_y.append(coords[hull.vertices[0], 1])

	# draw contour and white patch outside
	hull_points = list(zip(hull_points_x, hull_points_y))
	mask_outside_polygon(hull_points)
	plt.plot(hull_points_x, hull_points_y, 'k-')
	plt.axis("equal")
	plt.axis("off")


def main():

	create_voronoid()
	create_map(scale=0.08, offset=np.array([-0.2, 0.2]))

	# fix range of plot
	plt.xlim(- 2, 1.5)
	plt.ylim(- 1.5, 2)

	plt.savefig('experiment_setup.pdf')	
	# plt.show()


if __name__ == '__main__':
	main()