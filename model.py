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


def get_gauss_kernel(sigma, samples):
    """
    return a two dimesional gausian kernel of shape (size*(1/resolution),size*(1/resolution))
    with a std deviation of std
    """
    p = ny.ceil (2*ny.sqrt(2*ny.log(2))*sigma)
    r = ny.linspace(-p, p, samples)
    x,y = ny.meshgrid(r, r)
    b=bivariate_normal(x,y,sigma,sigma)
    A=(1/ny.max(b))
    B=A*b
    return x,y,B


#def get_gauss_kernel(sigma, size, res):
#    """
#    return a two dimesional gausian kernel of shape (size*(1/resolution),size*(1/resolution))
#    with a std deviation of std
#    """
#    x,y = ny.mgrid[-size/2:size/2:res,-size/2:size/2:res]
#    b=bivariate_normal(x,y,sigma,sigma)
#    A=(1/ny.max(b))
#    #A=1
#    return x,y,A*bivariate_normal(x, y, sigma, sigma)



class Stage(object):
    def __init__(self, Gx, C, N_number, N_angles):
        self.Gx = Gx
        self.C = C
        self.N_number=N_number
        self.N_angles=N_angles

    def do_v1(self, net_in, net_fb):
        v1_t= net_in + (self.C * net_fb * net_in)

        #v1_t /= ny.max(ny.absolute(v1_t))
        return v1_t


    def do_v2(self, v1_t):
        x = v1_t**2
        if not len (self.Gx):
            return x
        v2_t = ny.zeros_like(v1_t)
        for n in ny.arange(0, v1_t.shape[2]):
            v2_t[:,:,n] = convolve2d (v1_t[:,:,n], self.Gx, 'same',boundary='symm')

        #v2_t /= ny.max(v2_t)
        return v2_t


    def do_v3(self, v2_t):

        s=v2_t.shape
        N=ny.zeros_like(v2_t)
        M=N
        R=7  # CONSIDERED SURROUND (IN PIXEL) IN EACH DIRECTION

        # x,y IS THE AREA WHERE THE SURROUND R CAN BE CONSIDERED.
        a = ny.arange(R,s[0]-R)
        (x,y) = ny.meshgrid(a,a)
        (x,y) = (x.flatten(),y.flatten())

        # PURPOSE OF t1-t11: CREATES AN ARRAY (x1,y1) WHICH FORMS THE FRAME (THICKNESS=R) OUTSIDE THE AREA (x,y) WHERE THE SURROUND CANNOT BE CONSIDERED

        # t3,t4 CREATE AREA WITH X=RANGE(0:29) AND Y=RANGE(0,R). THIS AREA FORMS THE LOWER STRIP OF THE FRAME.
        t1=ny.arange(s[0])
        t2=ny.arange(R)
        t3,t4 = ny.meshgrid(t1,t2)
        t3,t4 = t3.flatten(),t4.flatten()

        # t6,t7 CREATE AREAS WITH X=RANGE(0:R, 29-R:29) AND Y=RANGE(R:29-R). THESE AREAS FORM THE SIDE STRIPS OF THE FRAME.
        t5 = ny.concatenate((ny.arange(0,R),ny.arange(s[0]-R, s[0])))
        t6,t7 = ny.meshgrid(t5,a)
        t6,t7 = t6.flatten(),t7.flatten()

        # t10,t11 CREATE AREAS WITH X=RANGE(0:29) AND Y=RANGE(29-R:29).THIS AREA FORMS THE UPPER STRIP OF THE FRAME.
        t9=ny.arange(s[0]-R, s[0])
        t10,t11 = ny.meshgrid(t1,t9)
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

                for k in ny.arange(0,self.N_number): # pixel-surround-average calculation per neuron (0-7)
                    N[x[i],y[i],k]=math.fsum(v2_t[m,n,k])
                    M[x[i],y[i],k]=N[x[i],y[i],k]/((R+R+1)**2)


            else:
                #for k in ny.arange(0,8):
                   # N[x1[i],y1[i],k]=v2_t[x1[i],y1[i],k]*((R+R+1)**2)
                M[x1[i],y1[i],:]=v2_t[x1[i],y1[i],:]

        v3_t = v2_t+v2_t+(v2_t-(1*M))#/ny.max(ny.absolute(v2_t-M))#+4.0)/7.0)*(v2_t-(0.3*M)))#(v2_t-(0.3*M))#(((ny.exp(-j/2.3)+0.00)/1.00)*(v2_t-(0.3*M)))

        v3_t /= (0.01+ny.max(ny.abs(v3_t)))

        return v3_t

    def do_all(self, net_in, net_fb, i, main_size, square_size, gauss_width, angle_outside, angle_inside, Channel):
        pop = pc.Population(main_size, square_size, gauss_width, self.N_number, self.N_angles, angle_outside, angle_inside)
        v1_t = self.do_v1(net_in, net_fb)
        v2_t = self.do_v2(v1_t)
        v3_t = self.do_v3(v2_t)


#        if i==8:
#            pp.figure(6+i-1)
#            if Channel==1:
#                pop.twoD_activation(v1_t, 0, i)
#                pop.twoD_activation(v2_t, 1, i)
#                pop.twoD_activation(v3_t, 2, i)
#            if Channel==2:
#                pop.twoD_activation(v1_t, 3, i)
#                pop.twoD_activation(v2_t, 4, i)
#                pop.twoD_activation(v3_t, 5, i)
        return v3_t


class Model(object):

    def __init__(self, cfg_file, feedback=True):

        self.do_feedback = feedback

        cfg = cp.RawConfigParser()

        cfg.read(cfg_file)

        self.main_size = cfg.getint('Input', 'main_size')
        self.square_size = cfg.getint('Input', 'square_size')
        self.time_frames = cfg.getint('Input', 'time_frames')
        self.angle_outside = cfg.getint('Input', 'angle_outside')
        self.angle_inside = cfg.getint('Input', 'angle_inside')
        self.gauss_width = cfg.getfloat('PopCode', 'neuron_sigma')
        self.N_number=cfg.getint('PopCode','number_of_neurons')
        N_angle=ny.zeros(self.N_number)
        for i in ny.arange(len(N_angle)):
            N_angle[i]=cfg.getfloat('PopCode','neuronal_tuning_angle_%i' %i)
        angle_num=ny.arange(0,self.N_number)
        self.N_angles=[N_angle[num] for num in angle_num]
        print self.N_angles

        v1_C = cfg.getfloat('V1', 'C')
        self.mt_kernel_sigma =  cfg.getfloat('MT', 'kernel_sigma')
        self.mt_kernel_samples =  cfg.getfloat('MT', 'kernel_samples')
        self.mt_kernel_size = cfg.getfloat('MT', 'kernel_size')
        self.mt_kernel_res = cfg.getfloat('MT', 'kernel_res')
        self.kernel_sigma1 =  cfg.getfloat('MT', 'kernel_sigma1')
        self.kernel_sigma2 =  cfg.getfloat('MT', 'kernel_sigma2')
        self.kernel_scale =  cfg.getfloat('MT', 'kernel_scale')

        self.x,self.y,self.mt_gauss = get_gauss_kernel(self.mt_kernel_sigma,self.mt_kernel_samples)# self.mt_kernel_size, self.mt_kernel_res)


        self.V1 = Stage([],v1_C, self.N_number, self.N_angles)
        self.MT = Stage(self.mt_gauss, 0, self.N_number, self.N_angles)


    def create_input(self):
        """
        definition of initial population codes for different time steps (is always the same one!!!)
        """
        j = pc.Population(self.main_size, self.square_size, self.gauss_width, self.N_number, self.N_angles, self.angle_outside, self.angle_inside)
        I= j.initial_pop_code()

        return I


    def run_model_full(self):
        self.input = self.create_input()
        pop = pc.Population(self.main_size, self.square_size, self.gauss_width, self.N_number, self.N_angles, self.angle_outside, self.angle_inside)
        X = ny.zeros((self.main_size, self.main_size, self.N_number, self.time_frames+1))
        FB=ny.zeros((self.main_size, self.main_size, self.N_number, self.time_frames+1))
        c=0
        for i in ny.arange(0,self.time_frames+1):
            if not i: # weil i=0 ist false
                X[:,:,:,i] = self.input
                if self.time_frames==0 and c==1:
                    pp.figure(5)
                    pop.twoD_activation(X[:,:,:,i], 0, i)

            else:
                inp = self.input
                v1= self.V1.do_all(inp, FB[:,:,:,i-1],i, self.main_size, self.square_size, self.gauss_width,self.angle_outside, self.angle_inside, 1)
                mt= self.MT.do_all(v1, 0, i, self.main_size, self.square_size, self.gauss_width,self.angle_outside, self.angle_inside, 2)
                X[:,:,:, i] = mt

                if self.do_feedback:
                    FB[:,:,:,i] = X[:,:,:,i]
            print i

        return X,FB


    def integrated_motion_direction(self):


        A = ny.zeros((self.main_size,self.main_size,self.time_frames+1))
        pop = pc.Population(self.main_size, self.square_size, self.gauss_width, self.N_number, self.N_angles, self.angle_outside, self.angle_inside)
        X,FB = self.run_model_full ()

        for i in ny.arange(0,self.time_frames+1):

            if i>0:

#                pp.figure(1)
#                pop.show_vectors(X[:,:,:,i],self.time_frames,i)
#
#                pp.figure(2)
#                pop.plot_pop(X[:,:,:,i],self.time_frames,i)
#
#                pp.figure(3)
#                pop.plot_cartesian_pop(X[:,:,:,i],self.time_frames,i)
#                if i>self.time_frames-3:
#                    pp.xlabel('Neuron number')

                pp.figure(4)
                pop.plot_row(1,15,X[:,:,:,i],i, self.time_frames)


            else:
#                pp.figure(1)
#                pop.show_vectors(X[:,:,:,i],self.time_frames,i)

#                pp.figure(2)
#                pop.plot_pop(X[:,:,:,i],self.time_frames,i)
#                #pp.suptitle('For $v^{3}=v^{2}+\\frac{1}{2}$ $5$ $(\mu=5\degree)$')
#                pp.suptitle('$For$ $v^{(3)}=\\frac{2 \cdot v^{(2)}+(v^{(2)}-M)}{max}$', fontsize=22)#'$For$ $v^{(3)}=\\frac{v^{(2)}+e^{(\\frac{-x}{2.3})}\cdot(v^{(2)}-(0.3\cdot M))}{max}$', fontsize=30)

#                pp.figure(3)
#                pop.plot_cartesian_pop(X[:,:,:,i],self.time_frames,i)
#

                pp.figure(4)
                pop.plot_row(1,15,X[:,:,:,i],i, self.time_frames)


#                pp.figure(3)
#                pop.show_vectors(self.input, self.time_frames,i)
#                pp.suptitle('For spatial kernel values:%s=%.2f and neuron kernel %s=%.2f.\n The y axis indicates the direction in color space.'\
#                % (u"\u03C3",self.mt_kernel_sigma,u"\u03C3", self.gauss_width) )

          #  print i

        fig=pp.figure(5)
        ax = fig.gca(projection='3d')
        surf=ax.plot_surface(self.x,self.y,self.mt_gauss, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
        pp.xlabel('Pixel')
        pp.ylabel('Pixel')
        pp.title('Spatial Kernel with %s=%.2f, samples=%.2f' % (u"\u03C3",self.mt_kernel_sigma, self.mt_kernel_samples))
        ax.set_zlim(ny.min(self.mt_gauss), ny.max(self.mt_gauss))
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        pp.colorbar(surf, shrink=0.5, aspect=5)


if __name__ == '__main__':

    M = Model('model.cfg') # feedback=True
    #M.show_mt_gauss()
    M.integrated_motion_direction()

    #M.run_model_full()
    pp.show()

