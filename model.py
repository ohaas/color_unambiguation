__author__ = 'ohaas'

import numpy as ny
import matplotlib.pyplot as pp
import pop_code as pc
from matplotlib.mlab import bivariate_normal
from scipy.signal import convolve2d
import ConfigParser as cp
import Neurons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math


def get_gauss_kernel(sigma, size, res):
    """
    return a two dimesional gausian kernel of shape (size*(1/resolution),size*(1/resolution))
    with a std deviation of std
    """
    x,y = ny.mgrid[-size/2:size/2:res,-size/2:size/2:res]
    b=bivariate_normal(x,y,sigma,sigma)
    A=(1/ny.max(b))
    B=A*bivariate_normal(x, y, sigma, sigma)
    #A=1
    return x,y,B
def gauss(sigma=0.75, x=ny.arange( 0.0, 8.0, 1), mu=0):
    return ny.exp(-(x-mu)**2/(2.0*sigma**2))


class Stage(object):
    def __init__(self, Gx, C):
        self.Gx = Gx
        self.C = C

    def do_v1(self, net_in, net_fb):
        v1_t= net_in + (self.C * net_fb * net_in)

        #v1_t= v1_t/ny.max(ny.absolute(v1_t))
        return v1_t

    def do_v2(self, v1_t):

        x = v1_t**2

        if not len (self.Gx):
            return x

        v2_t = ny.zeros_like (v1_t)
        for n in ny.arange(0, v1_t.shape[2]):
            v2_t[:,:,n] = convolve2d (x[:,:,n], self.Gx, 'same')

        v2_t= v2_t/ny.max(ny.absolute(v2_t))
        return v2_t

    def do_v3(self, v2_t, j):

        s=v2_t.shape
        N=ny.zeros_like(v2_t)
        M=N
        R=7  # CONSIDERED SURROUND IN EACH DIRECTION, PER PIXEL

        # x,y IS THE AREA WHERE THE SURROUND R CAN BE CONSIDERED.
        a = ny.arange(R,s[0]-R)
        (x,y) = ny.meshgrid(a,a)
        (x,y) = (x.flatten(),y.flatten())

        # PURPOSE OF t1-t11: CREATES AN ARRAY (x1,y1) WHICH FORMS THE FRAME (THICKNESS=R) OUTSIDE THE AREA (x,y) WHERE THE SURROUND CAN BE CONSIDERED
        t1=ny.arange(s[0])
        t2=ny.arange(R)
        t3,t4 = ny.meshgrid(t1,t2)
        t3,t4 = t3.flatten(),t4.flatten()

        t5 = ny.concatenate((ny.arange(0,R),ny.arange(s[0]-R, s[0])))
        t6,t7 = ny.meshgrid(t5,a)
        t6,t7 = t6.flatten(),t7.flatten()

        t8=ny.arange(s[0])
        t9=ny.arange(s[0]-R, s[0])
        t10,t11 = ny.meshgrid(t8,t9)
        t10,t11 = t10.flatten(),t11.flatten()

        # ZEROS IN THE BEGINNING, SO THAT INDEX i IS AT THE CORRECT POSITION IN x1/y1 AFTER SURROUND-AVERAGE CALCULATION
        x1 = ny.concatenate((ny.tile(0.0,len(x)),t3,t6,t10))
        y1 = ny.concatenate((ny.tile(0.0,len(x)),t4,t7,t11))

        for i in ny.arange(len(x1)):
            if i in ny.arange(len(x)):

                c1 = ny.arange(x[i]-R,x[i]+R+1)
                c2 = ny.arange(y[i]-R,y[i]+R+1)
                (m,n)=ny.meshgrid(c1,c2)
                m=m.flatten()
                n=n.flatten()
                for k in ny.arange(0,8): # pixel-surround-average calculation per neuron (0-7)
                    N[x[i],y[i],k]=ny.sum(v2_t[m,n,k], axis=0)
                    M[x[i],y[i],k]=N[x[i],y[i],k]/((R+R+1)**2)
            else:
                for k in ny.arange(0,8):
                    M[x1[i],y1[i],k]=v2_t[x1[i],y1[i],k]/((R+R+1)**2)#ny.max(v2_t[x1[i],y1[i],k])
        v3_t = v2_t+(v2_t-(0.5*M))#+4.0)/7.0)*(v2_t-(0.3*M)))#(v2_t-(0.3*M))#(((ny.exp(-j/2.3)+0.00)/1.00)*(v2_t-(0.3*M)))

        v3_t = v3_t/ny.max(ny.absolute(v3_t))
        return v3_t

    def do_all(self, net_in, net_fb, i):
        v1_t = self.do_v1(net_in, net_fb)
        v2_t = self.do_v2(v1_t)
        v3_t = self.do_v3(v2_t, i)
        return v1_t,v2_t,v3_t


class Model(object):

    def __init__(self, cfg_file, feedback=True):

        self.do_feedback = feedback

        cfg = cp.RawConfigParser()

        cfg.read(cfg_file)

        self.main_size = cfg.getint('Input', 'main_size')
        self.square_size = cfg.getint('Input', 'square_size')
        self.time_frames = cfg.getint('Input', 'time_frames')
        self.gauss_width = cfg.getfloat('PopCode', 'neuron_sigma')

        v1_C = cfg.getfloat('V1', 'C')
        self.mt_kernel_sigma =  cfg.getfloat('MT', 'kernel_sigma')
        self.mt_kernel_size =  cfg.getfloat('MT', 'kernel_size')
        self.mt_kernel_res =  cfg.getfloat('MT', 'kernel_res')

        self.x,self.y,self.mt_gauss = get_gauss_kernel(self.mt_kernel_sigma, self.mt_kernel_size, self.mt_kernel_res)


        self.V1 = Stage([], v1_C)
        self.MT = Stage(self.mt_gauss, 0)


    def create_input(self):
        """
        definition of initial population codes for different time steps (is always the same one!!!)
        """
        j = pc.Population(self.main_size, self.square_size, self.gauss_width)
        I= j.initial_pop_code()

        return I


    def run_model_full(self, i):
        self.input = self.create_input()
        pop = pc.Population(self.main_size, self.square_size, self.gauss_width)
        X = ny.zeros((self.main_size, self.main_size, 8, self.time_frames+1))
        FB=ny.zeros((self.main_size, self.main_size, 8, self.time_frames+1))
        c=0
        if i==0:
            X[:,:,:,i] = self.input
            if self.time_frames==0 and c==1:
                pp.figure(5)
                pop.twoD_activation(X[:,:,:,i], 0, i)

        else:
            inp = self.input

            v1_a, v1_b, v1_c = self.V1.do_all(inp, FB[:,:,:,i-1],i)
            if c==1:
                pp.figure(5+i-1)
                pop.twoD_activation(v1_a, 0, i)
                pop.twoD_activation(v1_b, 1, i)
                pop.twoD_activation(v1_c, 2, i)

            mt_a, mt_b, mt_c = self.MT.do_all(v1_c, 0, i)
            if c==1:
                pp.figure(5+i-1)
                pop.twoD_activation(mt_a, 3, i)
                pop.twoD_activation(mt_b, 4, i)
                pop.twoD_activation(mt_c, 5, i)
            X[:,:,:, i] = mt_c

            if self.do_feedback:
                FB[:,:,:,i] = X[:,:,:,i]


        return X,FB


    def integrated_motion_direction(self):


        A = ny.zeros((self.main_size,self.main_size,self.time_frames+1))

        for i in ny.arange(0,self.time_frames+1):

            pop = pc.Population(self.main_size, self.square_size, self.gauss_width)
            X,FB = self.run_model_full (i)

            if i>0:

#                pp.figure(2)
#                pop.show_vectors(X[:,:,:,i],self.time_frames,i)

                pp.figure(1)
                pop.plot_pop(X[:,:,:,i],self.time_frames,i)

                pp.figure(2)
                pop.plot_cartesian_pop(X[:,:,:,i],self.time_frames,i)

#                pp.figure(3)
#                pop.heat_map(X[:,:,:,i], i, self.time_frames, A)

                pp.figure(4)
                pop.plot_row(1,15,X[:,:,:,i],i, self.time_frames)

#                pp.figure(5)
#                pop.twoD_activation(X[:,:,:,i], i, self.time_frames)

            else:
                pp.figure(1)
                pop.plot_pop(X[:,:,:,i],self.time_frames,i)
                #pp.suptitle('For $v^{3}=v^{2}+\\frac{1}{2}$ $5$ $(\mu=5\degree)$')
                pp.suptitle('$For$ $v^{(3)}=v^{(2)}+\\frac{e^{(\\frac{-x}{2.3})}\cdot(v^{(2)}-(0.3\cdot M))}{max}$', fontsize=30)

                pp.figure(2)
                pop.plot_cartesian_pop(X[:,:,:,i],self.time_frames,i)

#                pp.figure(3)
#                pop.heat_map(X[:,:,:,i], i, self.time_frames, A)

                pp.figure(4)
                pop.plot_row(1,15,X[:,:,:,i],i, self.time_frames)

#                pp.figure(5)
#                pop.twoD_activation(X[:,:,:,i], i, self.time_frames)

#                pp.figure(2)
#                pop.show_vectors(self.input[:,:,:,i], self.time_frames,i)
#                pp.suptitle('For spatial kernel values:%s=%.2f, size=%.2f, res=%.2f and neuron kernel %s=%.2f.\n The y axis indicates the direction in color space.'\
#                % (u"\u03C3",self.mt_kernel_sigma, self.mt_kernel_size, self.mt_kernel_res,u"\u03C3", self.gauss_width) )

            print i

#        fig=pp.figure(2)
#        ax = fig.gca(projection='3d')
#        surf=ax.plot_surface(self.x,self.y,self.mt_gauss, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
#        pp.xlabel('Pixel')
#        pp.ylabel('Pixel')
#        pp.title('Spatial Kernel with %s=%.2f, size=%.2f, res=%.2f' % (u"\u03C3",self.mt_kernel_sigma, self.mt_kernel_size, self.mt_kernel_res))
#        ax.set_zlim(ny.min(self.mt_gauss), ny.max(self.mt_gauss))
#        ax.zaxis.set_major_locator(LinearLocator(10))
#        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
#        pp.colorbar(surf, shrink=0.5, aspect=5)


if __name__ == '__main__':

    M = Model('model.cfg') # feedback=True
    #M.show_mt_gauss()
    M.integrated_motion_direction()

    #M.run_model_full()
    pp.show()

