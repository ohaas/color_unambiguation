__author__ = 'ohaas'

import numpy as ny
import matplotlib.pyplot as pp
from pylab import *
from distutils.filelist import findall



#y=ny.exp(-(x/2.8))
#y1=(ny.exp(-x/2.3)+0.02)/1.02
#y2=ny.exp(-(x/2.0)+1)
##print ny.exp(-(10/2.5))
#pp.plot(x,y)
#pp.plot(x,y1)
##pp.plot(x,y1)
##pp.plot(x,y2)
#pp.show()


#fig = pp.figure()
#im  = pp.imread('colorspace_polar_z=0.0.jpg')
#image=imshow(im,origin='centre')
#pp.plot(x,x)
#ax2 = pp.subplot(111,projection='polar')
#ax2.axes.get_xaxis().set_visible(False)
#ax2.axes.get_yaxis().set_visible(False)
#ax2.patch.set_facecolor('none')





# get current axes object
#frame1 = pp.gca()
# get current figure
fig = pp.gcf()
# read the image file
frame1 = fig.add_subplot(111, frameon=False)

pic = pp.imread('colorspace_polar_z=0.0.jpg')
# the picture is upside down so rotate and fip it
pic = np.fliplr(ny.rot90(pic, k=2))
# draw it on the canvas
pp.imshow(pic, figure=frame1)
# hide axes
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)
frame1.set_axis_off()


frame2 = fig.add_subplot(111, polar=True, axisbg='none', frameon=False)
r = np.arange(0, 1.0, 0.01)
theta = 2*ny.pi*r
frame2.plot(theta,r,'k')
frame2.axes.get_xaxis().set_visible(False)
frame2.axes.get_yaxis().set_visible(False)




#ax = pp.subplot(111)
#
#im=pp.imread('colorspace_polar_z=0.0.jpg')
#im = np.fliplr(np.rot90(im, k=2))
#pp.imshow(im)
## hide axes
#frame1.axes.get_xaxis().set_visible(False)
#frame1.axes.get_yaxis().set_visible(False)
#
##pp.draw()
##ax = fig.add_subplot(111, polar=True)
##ax.plot(x,x,'k')
#ax.axes.get_xaxis().set_visible(False)
#ax.axes.get_yaxis().set_visible(False)
#ax.patch.set_facecolor('none')

#ax.pcolormesh(im) #X,Y & data2D must all be same dimensions
#ax = fig.add_subplot(111)

#ax = fig.add_subplot(111, polar=True)
#pp.scatter(x, x, 'k')
#fig.frameon= False
#pp.subplot(axisbg='r', polar=True)
#pp.plot(x,x,'k')
#fig=pp.polar(x,x,'k')
#fig.patch.set_facecolor('red')

pp.show()
# function defined in polar coordinate
#
#N = 150
#r = 2*ny.random.rand(N)
#theta = 2*ny.pi*ny.random.rand(N)
#area = 200*r**2*ny.random.rand(N)
#colors = theta
