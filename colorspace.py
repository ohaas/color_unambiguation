__author__ = 'ohaas'


import numpy as ny
import matplotlib.pyplot as pp
from matplotlib import rc

res=1.0/60.0
i=1
for r in ny.arange(0.0,1.0+res,res):
    res1=res*(1/(1*(r+res)))
    for t in ny.arange(0.0,2*ny.pi, res1):
        x=ny.cos(t)*r
        y=ny.sin(t)*r
        z=2.0
        maxi=1+(z/2.0)
        L=(((z+x)/2.0)/maxi)
        M=(((z-x)/2.0)/maxi)
        S=((y+(z/2.0))/maxi)
        #print L,M,S
        all=((2*ny.pi)/res1)*((1.0+res)/res)
        print i ,' th trial of the total', all
        i+=1
        pp.scatter(x, y, color=(L,M,S), edgecolors='none')
        pp.axis('equal')
        ax1 = pp.axes(frameon=False)
        ax1.axes.get_yaxis().set_visible(False)
        ax1.axes.get_xaxis().set_visible(False)
        pp.axhline(xmin=0.1, xmax=0.88, linewidth=0.7, color='grey', linestyle='--')
        pp.axvline(ymin=0.04, ymax=0.97,linewidth=0.7, color='grey', linestyle='--')
        rc('text', usetex=True)
        pp.annotate('$S-M$', xy=(1.2,0), verticalalignment='center', fontsize=20)
        pp.annotate(r"$\displaystyle S-\frac{L+M}{2}$", xy=(0,1.1), horizontalalignment='center', fontsize=20)
pp.show()



#for S in ny.arange(0.0,1.0,res):
#    for M in ny.arange(0.0,1.0,res):
#        for L in ny.arange(0.0,1.0,res):
#            print L,M,S
#            x=L-M
#            y=S-((L+M)/2)
#            pp.scatter(x,y,color=(L,M,S))