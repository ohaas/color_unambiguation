__author__ = 'ohaas'

from matplotlib import cm
from matplotlib import colors
import numpy as ny
import matplotlib.pyplot as pp

cdict = {'red':   [(0.0,  0.0, 0.0),(0.5,  1.0, 1.0),(1.0,  1.0, 1.0)],'green': [(0.0,  0.0, 0.0),(0.25, 0.0, 0.0),(0.75, 1.0, 1.0),(1.0,  1.0, 1.0)],'blue':  [(0.0,  0.0, 0.0),(0.5,  0.0, 0.0),(1.0,  1.0, 1.0)]}
my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict)
x=ny.arange(0,361)
y=1
pp.pcolor(x,y, cmap=my_cmap)