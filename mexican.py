__author__ = 'ohaas'

import matplotlib
matplotlib.use("Agg")
import numpy as ny
import matplotlib.pyplot as pp
import model
from scipy.signal import convolve2d
import pop_code as pc
import ConfigParser as cp
import math
from matplotlib.mlab import bivariate_normal
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

def get_gauss_kernel(sigma, samples):
    """
    return a two dimesional gausian kernel of shape (size*(1/resolution),size*(1/resolution))
    with a std deviation of std
    """
    p = ny.ceil (2*ny.sqrt(2*ny.log(2))*sigma)
    r = ny.linspace(-p, p, samples)
    x,y = ny.meshgrid(r, r)
    b=bivariate_normal(x,y,sigma,sigma)
    A=(1/ny.sum(b))
    B=A*b
    return x,y,B

class Mexican_hat(object):
    def __init__(self, cfg_file):

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
        self.mt_kernel_sigma =  cfg.getfloat('MT', 'kernel_sigma')
        self.kernel_sigma1 =  cfg.getfloat('MT', 'kernel_sigma1')
        self.kernel_sigma2 =  cfg.getfloat('MT', 'kernel_sigma2')
        self.kernel_samples =  cfg.getfloat('MT', 'kernel_samples')
        self.kernel_scale =  cfg.getfloat('MT', 'kernel_scale')
        angle=self.N_angles
        self.vec = ny.matrix([[ny.cos(fi*(ny.pi/180)), ny.sin(fi*(ny.pi/180))] for fi in angle])
        self.x,self.y,self.mt_gauss = get_gauss_kernel(self.mt_kernel_sigma, self.kernel_samples)


    def get_dog_kernel(self, sigma1, sigma2, samples):

        # return a 2d DoG Kernel

        x1,y1,DoG1 = get_gauss_kernel(sigma1, samples)
        x2,y2,DoG2 = get_gauss_kernel(sigma2, samples)
        A=1
        DoG=A*(DoG1 - (self.kernel_scale*DoG2))
        return DoG/ny.max(ny.abs(DoG))

    def create_input(self, angle_outside, angle_inside):
        """
        definition of initial population codes for different time steps (is always the same one!!!)
        """
        j = pc.Population(self.main_size, self.square_size, self.gauss_width, self.N_number, self.N_angles, angle_outside, angle_inside)
        I= j.initial_pop_code()
        return I

    def do_v1(self, net_in, net_fb, C):
        v1_t= net_in + (C*net_fb * net_in)

        #v1_t /= ny.max(ny.absolute(v1_t))
        return v1_t

    def do_v2(self, v1_t, Gx, iteration):

        x = v1_t#**2
        v2_t = ny.zeros_like(v1_t)
        y = ny.zeros_like(v1_t)

        if not len (Gx):
            for n in ny.arange(0, v1_t.shape[2]):
                #y[:,:,n]= x[:,:,n]
                self.DoG=self.get_dog_kernel(self.kernel_sigma1, self.kernel_sigma2, self.kernel_samples)
                y[:,:,n]= convolve2d (x[:,:,n], self.DoG, 'same',boundary='symm')

            return y

        #for a in ny.arange(0,self.main_size):
         #   v2_t[a,:,:] = convolve2d (x[a,:,:], self.get_dog_kernel(), 'same',boundary='symm')
        #for b in ny.arange(0,self.main_size):
         #   v2_t[:,b,:] = convolve2d (x[:,b,:], Gx, 'same',boundary='symm')
        for n in ny.arange(0, v1_t.shape[2]):
            v2_t[:,:,n] = x[:,:,n]
            #v2_t[:,:,n] = convolve2d (x[:,:,n], self.get_dog_kernel(self.kernel2_sigma1, self.kernel2_sigma2, self.kernel2_samples), 'same',boundary='symm')
            #v2_t[:,:,n] = convolve2d (x[:,:,n], Gx, 'same',boundary='symm')
            #v2_t[:,:,n] = convolve2d(convolve2d (x[:,:,n], Gx, 'same',boundary='symm'), self.get_dog_kernel(self.kernel2_sigma1, self.kernel2_sigma2, self.kernel2_samples), 'same',boundary='symm')

        #v2_t/=ny.max(ny.absolute(v2_t))
        return v2_t


    def do_all(self, net_in, net_fb, gauss, last, iteration, C):
        v2_t = self.do_v2(net_in, gauss, iteration)
        return v2_t/ny.max(v2_t)

    def max_colour_induced_shift(self, color_space_step_size, angle_outside):

        if angle_outside-180>0:
            angle_inside = ny.concatenate((ny.arange(angle_outside-180,360,color_space_step_size),ny.arange(0,angle_outside-180+color_space_step_size,color_space_step_size)))
        else:
            angle_inside = ny.concatenate((ny.arange(angle_outside+180,360,color_space_step_size),ny.arange(0,angle_outside+180+color_space_step_size,color_space_step_size)))

        rel_angle_inside=ny.arange(-180, 180+color_space_step_size, color_space_step_size)
        POP=ny.zeros((self.main_size,self.main_size,self.N_number,2*len(angle_inside)))
        Angles=ny.zeros((self.main_size,self.main_size,2*len(angle_inside)))
        Shift=ny.zeros((self.main_size,self.main_size,len(angle_inside)))

        max_shift = ny.zeros(len(rel_angle_inside))
        extrema_shift = ny.zeros(len(rel_angle_inside))
        min_shift = ny.zeros(len(rel_angle_inside))



        s=POP.shape
        before=ny.arange(0,2*len(angle_inside),2)

        for i, a in enumerate(angle_inside):
            print a
            POP[:,:,:,before[i]]=self.create_input(angle_outside, a)
            #After=self.run_convolution(POP[:,:,:,before[i]])
            POP[:,:,:,before[i]+1]=self.run_convolution(POP[:,:,:,before[i]], angle_inside, angle_outside)
            #self.do_v2(POP[:,:,:,before[i]],[])#After[:,:,:,self.time_frames]#self.do_v2(POP[:,:,:,before[i]], self.mt_gauss)

        for j in ny.arange(0,s[3]):
            for x in ny.arange(0,self.main_size):
                for y in ny.arange(0,self.main_size):
                    multiple=ny.multiply(POP[x,y,:,j],ny.transpose(self.vec))

                    x1=ny.sum(multiple[0,:])
                    y1=ny.sum(multiple[1,:])

                    if x1<0:
                        Angles[x,y,j]=(ny.arctan(y1/x1)+ny.pi)
                    elif x1>0 and y1>=0:
                        Angles[x,y,j]=ny.arctan(y1/x1)
                    elif x1>0 and y1<0:
                        Angles[x,y,j]=(ny.arctan(y1/x1)+(2*ny.pi))
                    elif x1==0 and y1>0:
                        Angles[x,y,j]=ny.pi/2.0
                    else:
                        Angles[x,y,j]=3*ny.pi/2
        for n in ny.arange(0,len(rel_angle_inside)):
            Shift[:,:,n]=((Angles[:,:,before[n]+1])*180/ny.pi)-((Angles[:,:,before[n]])*180/ny.pi)
            extrema_shift[n]=(ny.max((Angles[:,:,before[n]+1])*180/ny.pi))-(ny.max((Angles[:,:,before[n]])*180/ny.pi))
            min_shift[n]=(ny.min((Angles[:,:,before[n]+1])*180/ny.pi))-(ny.min((Angles[:,:,before[n]])*180/ny.pi))

            for x in ny.arange(0,self.main_size):
                for y in ny.arange(0,self.main_size):
                    if Shift[x,y,n]>180:
                        Shift[x,y,n]=(Shift[x,y,n]-360)
                    elif Shift[x,y,n]<-180:
                        Shift[x,y,n]=(360+Shift[x,y,n])
            if extrema_shift[n]>180:
                extrema_shift[n]=(extrema_shift[n]-360)
            elif extrema_shift[n]<-180:
                extrema_shift[n]=(360+extrema_shift[n])
            elif min_shift[n]>180:
                min_shift[n]=(min_shift[n]-360)
            elif min_shift[n]<-180:
                min_shift[n]=(360+min_shift[n])
            if abs(min_shift[n])>abs(extrema_shift[n]):
                max_shift[n]=min_shift[n]
            elif abs(extrema_shift[n])>abs(min_shift[n]):
                max_shift[n]=extrema_shift[n]
            x=ny.where(abs(Shift[:,:,n])==ny.max(abs(Shift[:,:,n])))
            max_shift[n]=Shift[x[0][0],x[1][0],n]

        return rel_angle_inside, max_shift



        #####
    def plot_shifts(self, angle_outside, rel_angle_inside, max_shift):
        fig=pp.figure()

        ax = fig.add_subplot(111)
        C = ny.genfromtxt("Colormatrix.txt")
        t =(angle_outside*(2*ny.pi/360))/(1/60.0)
        cm = C[t, :]
        str_list = []
        for i in ny.arange(0,len(self.N_angles)):
            str_list.append('$neuron$ $%i$ $(\mu=%.1f\degree)$\n' %(i+1,self.N_angles[i]))
        List=''.join(str_list)
        ax.scatter(rel_angle_inside,max_shift, c=cm, lw = 0, label=List)

        # Shink current axis by 20%
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

        # Put a legend to the right of the current axis
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        pp.grid(True)
        pp.xticks(ny.arange(-180,230,45))
        pp.ylim(-30,30,10)
        pp.xlabel('Relative angle in colour space for stimulus [$^\circ$], with a background color of %d$^\circ$. \n Convolution ' \
                  'kernel uses $\sigma1=$ %.1f and $\sigma2=$ %.1f with %.1f samples and a scaling of %.1f.'%(angle_outside, self.kernel_sigma1, self.kernel_sigma2, self.kernel_samples, self.kernel_scale))
        pp.ylabel('maximal shift [degree]')
        pp.title('For %i neurons with %i $\sigma$' %(self.N_number,self.gauss_width))
        #            pp.figure(2+n)
        #            pp.imshow(Angles[:,:,before[n]]*180/ny.pi)
        #            pp.title('colour inside = %i degree' %(n*90))
        #            pp.colorbar()
        return fig


    def run_convolution(self, input, angle_inside, angle_outside):

        if not len(input):
            input=self.create_input(angle_outside, angle_inside)

        POP1 = ny.zeros((self.main_size,self.main_size,self.N_number,self.time_frames+1))
        FB = ny.zeros((self.main_size, self.main_size, self.N_number, self.time_frames+1))

        POP1[:,:,:,0]=input
        POP1[:,:,:,1]=self.do_all(input, FB[:,:,:,0], [], input, 1, 1.0)
        return POP1[:,:,:,self.time_frames]

#     pop = pc.Population(self.main_size, self.square_size, self.gauss_width, self.N_number, self.N_angles, self.angle_outside, self.angle_inside)
    #        for i in ny.arange(0,self.time_frames+1):
#            print i
#
#            if i>0:
#                if i==1:
#                    POP1[:,:,:,i]=self.do_all(input, FB[:,:,:,i-1], [], input, i, 1.0)
#                    #POP1[:,:,:,i]=self.do_all(POP[:,:,:,i], 0, self.mt_gauss,input, i, 0)#self.do_v2(POP[:,:,:,i],self.mt_gauss)
#                    FB[:,:,:,i] = POP1[:,:,:,i]
#                else:
#                    POP1[:,:,:,i]=input #self.do_all(input, FB[:,:,:,i-1], [], input, i, 1.0)
#                    #POP1[:,:,:,i]=self.do_all(POP[:,:,:,i], 0, self.mt_gauss,input,i, 0)#self.do_v2(POP[:,:,:,i],self.mt_gauss)
#                    FB[:,:,:,i] = POP1[:,:,:,i]

#            pp.figure(1)
#            pop.show_vectors(POP1[:,:,:,i], self.time_frames, i)
#            pp.figure(1)
#            pop.plot_pop(POP1[:,:,:,i],self.time_frames,i)
#            pp.figure(2)
#            pop.plot_cartesian_pop(POP1[:,:,:,i],self.time_frames,i)
#            pp.figure(3)
#            pop.plot_row(1,15,POP1[:,:,:,i],i, self.time_frames)
#            pp.figure(4)
#            pop.twoD_activation(POP[:,:,:,i], i, self.time_frames)
#        fig=pp.figure(5)
#        ax=fig.gca(projection='3d')
#        surf=ax.plot_surface(self.x,self.y,self.DoG, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
#        pp.xlabel('Pixel')
#        pp.ylabel('Pixel')
#        pp.suptitle('Mexican-hat Kernel with %s1=%.1f, %s2=%.1f, samples=%.1f and scale=%.2f' % (u"\u03C3",self.kernel_sigma1, u"\u03C3",self.kernel_sigma2, self.kernel_samples, self.kernel_scale))
#        ax.set_zlim(ny.min(self.DoG), ny.max(self.DoG))
#        ax.zaxis.set_major_locator(LinearLocator(10))
#        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        #pp.colorbar(surf, shrink=0.5, aspect=5)

       # return POP1[:,:,:,self.time_frames]

if __name__ == '__main__':
    M = Mexican_hat('model.cfg')
    for angle_outside in ny.arange(0, 360, 45.0):
        angle, shift = M.max_colour_induced_shift(22.5, angle_outside)
        fig = M.plot_shifts(angle_outside, angle, shift)
        fig.savefig('shift_bg_%4.1f.png' % angle_outside)

    #M.run_convolution([])
    #pp.show()


