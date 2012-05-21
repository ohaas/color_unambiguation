__author__ = 'ohaas'

__author__ = 'ohaas'


import numpy as ny
import matplotlib.pyplot as pp
from matplotlib import rc


class twoD(object):

    def __init__(self, res, z):
        self.res=res
        self.z=z

    def circle(self):

    #        i=1
        for r in ny.arange(0.0,1.0+self.res,self.res):
            res1=self.res*(1/(r+self.res))
            t = ny.arange(0.0,2*ny.pi, res1)
            x=ny.cos(t)*r
            y=ny.sin(t)*r
            maxi=1+(ny.absolute(self.z)/2.0)
            L=((self.z+x)/2.0)
            M=((self.z-x)/2.0)
            S=(y+(self.z/2.0))
            #                print L,M,S
            #                all=((2*ny.pi)/res1)*((1.0+self.res)/self.res)
            #                print i ,' th trial of the total', all
            #                i+=1
            c=ny.array((L,M,S))
            c=0.5+(0.5*c/maxi)
            pp.scatter(x, y, color=c.T, edgecolors='none')

            pp.axis('equal')
            ax1 = pp.axes(frameon=False)
            ax1.axes.get_yaxis().set_visible(False)
            ax1.axes.get_xaxis().set_visible(False)
            pp.axhline(xmin=0.1, xmax=0.88, linewidth=0.7, color='grey', linestyle='--')
            pp.axvline(ymin=0.04, ymax=0.97,linewidth=0.7, color='grey', linestyle='--')
            rc('text', usetex=True)
            pp.annotate('$S-M$', xy=(1.2,0), verticalalignment='center', fontsize=20)
            pp.annotate(r"$\displaystyle S-\frac{L+M}{2}$", xy=(0,1.1), horizontalalignment='center', fontsize=20)

    def hexagon(self):

        l =len(ny.arange(0.0,1.0,self.res))
        a = ny.array((ny.arange(0.0,1.0,self.res)))
        S = ny.repeat(a,l**2)
        M = ny.tile(ny.repeat(a,l),l)
        L = ny.tile(ny.tile(a,l),l)
        c=ny.array((L,M,S))
        x=L-M
        y=S-((L+M)/2)
        pp.scatter(x,y,color=c.T)

        pp.axis('equal')
        pp.gca().axison = False
        pp.axhline(xmin=0.1, xmax=0.88, linewidth=0.7, color='grey', linestyle='--')
        pp.axvline(ymin=0.04, ymax=0.97,linewidth=0.7, color='grey', linestyle='--')
        rc('text', usetex=True)
        pp.annotate('$S-M$', xy=(1.2,0), verticalalignment='center', fontsize=20)
        pp.annotate(r"$\displaystyle S-\frac{L+M}{2}$", xy=(0,1.1), horizontalalignment='center', fontsize=20)


if __name__ == '__main__':
    twoD(1.0/70.0, 0.0).hexagon()
    pp.show()