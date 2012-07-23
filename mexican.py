__author__ = 'ohaas'

import numpy as ny
import matplotlib.pyplot as pp
import model
from scipy.signal import convolve2d
import pop_code as pc
import ConfigParser as cp

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
        self.kernel_sigma1 =  cfg.getfloat('MT', 'kernel_sigma1')
        self.kernel_sigma2 =  cfg.getfloat('MT', 'kernel_sigma2')
        self.kernel_samples =  cfg.getfloat('MT', 'kernel_samples')
        self.kernel_scale =  cfg.getfloat('MT', 'kernel_scale')
        self.vec = ny.matrix(((1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)))

    def get_dog_kernel(self):

        # return a 2d DoG Kernel

        x1,y1,DoG1 = model.get_gauss_kernel(self.kernel_sigma1, self.kernel_samples)
        x2,y2,DoG2 = model.get_gauss_kernel(self.kernel_sigma2, self.kernel_samples)
        A=10
        return A*(DoG1 - (self.kernel_scale*DoG2))

    def create_input(self, angle_outside, angle_inside):
        """
        definition of initial population codes for different time steps (is always the same one!!!)
        """
        j = pc.Population(self.main_size, self.square_size, self.gauss_width, angle_outside, angle_inside)
        I= j.initial_pop_code()
        return I


    def do_v2(self, pop_code):

        v2_t = ny.zeros_like(pop_code)
        for n in ny.arange(0, pop_code.shape[2]):
            v2_t[:,:,n] = convolve2d (pop_code[:,:,n], self.get_dog_kernel(), 'same',boundary='symm')

        v2_t /= ny.max(v2_t)
        return v2_t

    def max_colour_induced_shift(self, color_space_step_size):
        angle_inside = ny.arange(0,360,color_space_step_size)
        rel_angle_inside=ny.arange(-180, 180+color_space_step_size, color_space_step_size)
        POP=ny.zeros((self.main_size,self.main_size,8,2*len(angle_inside)))
        Angles=ny.zeros((self.main_size,self.main_size,2*len(angle_inside)))
        Shift=ny.zeros((self.main_size,self.main_size,len(angle_inside)))

        max_shift = ny.zeros(len(angle_inside))

        s=POP.shape
        before=ny.arange(0,2*len(angle_inside),2)
        i=0
        for a in angle_inside:
            POP[:,:,:,before[i]]=self.create_input(self.angle_outside, a)
            POP[:,:,:,before[i]+1]=self.do_v2(POP[:,:,:,before[i]])
            i+=1
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
        for n in ny.arange(0,len(angle_inside)):
            Shift[:,:,n]=((Angles[:,:,before[n]+1])*180/ny.pi)-((Angles[:,:,before[n]])*180/ny.pi)
            for x in ny.arange(0,self.main_size):
                for y in ny.arange(0,self.main_size):
                    if Shift[x,y,n]>180:
                        Shift[x,y,n]=(Shift[x,y,n]-360)
                    elif Shift[x,y,n]<-180:
                        Shift[x,y,n]=(360+Shift[x,y,n])
            x=ny.where(abs(Shift[:,:,n])==ny.max(abs(Shift[:,:,n])))
            max_shift[n]=Shift[x[0][0],x[1][0],n]
            pp.figure(1)
            if n in ny.arange(len(max_shift)-(45/color_space_step_size),len(max_shift)):
                pp.scatter(rel_angle_inside[n-(len(max_shift)-(45/color_space_step_size))],max_shift[n])

            elif not n:
                pp.scatter(rel_angle_inside[len(rel_angle_inside)-1], max_shift[len(max_shift)-1])
                pp.scatter(rel_angle_inside[n+(45/color_space_step_size)],max_shift[n])

            else:
                pp.scatter(rel_angle_inside[n+(45/color_space_step_size)],max_shift[n])

            pp.grid(True)
            pp.xticks(ny.arange(-180,230,45))
            pp.xlabel('Angle in colour space for stimulus [degree], with a background color of 135 degree')
            pp.ylabel('maximal shift [degree]')
        #            pp.figure(2+n)
        #            pp.imshow(Angles[:,:,before[n]]*180/ny.pi)
        #            pp.title('colour inside = %i degree' %(n*90))
        #            pp.colorbar()


    def run_convolution(self):

        POP=ny.zeros((self.main_size,self.main_size,8,self.time_frames+1))
        POP[:,:,:,0]=self.create_input(self.angle_outside, self.angle_inside)
        pop = pc.Population(self.main_size, self.square_size, self.gauss_width, self.angle_outside, self.angle_inside)


        for i in ny.arange(0,self.time_frames+1):
            print i

            if i>0:
                POP[:,:,:,i]=self.do_v2(POP[:,:,:,i-1])
            pp.figure(1)
            pop.plot_pop(POP[:,:,:,i],self.time_frames,i)
            pp.figure(2)
            pop.plot_cartesian_pop(POP[:,:,:,i],self.time_frames,i)
            pp.figure(3)
            pop.plot_row(1,15,POP[:,:,:,i],i, self.time_frames)
            pp.figure(4)
            pop.twoD_activation(POP[:,:,:,i], i, self.time_frames)




if __name__ == '__main__':
    M=Mexican_hat('model.cfg')
    #M.run_convolution()
    M.max_colour_induced_shift(1)
    pp.show()


