__author__ = 'ohaas'

import numpy as ny
import matplotlib as m
import matplotlib.pyplot as pp
from matplotlib.patches import FancyArrowPatch as fap
import Stimulus
import Neurons
import math
import Image, ImageDraw
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid

class Population(object):


    def __init__(self, main_size, square_size, gauss_width, N_number, N_angles, angle_outside, angle_inside):
        """
        main_size= IMAGE SIZE SQUARED (main_size x main_size), square_size=STIMULUS SIZE SQUARED,
        start=STIMULUS STARTING POINT SQUARED (lower left stimulus corner), gauss_width= WIDTH OF NEURONAL GAUSS TUNING CURVE
        """
        self.main_size = main_size
        self.square_size = square_size
        self.I = Stimulus.image(main_size,square_size)
        self.width=gauss_width
        self.N_number=N_number
        self.N_angles=N_angles


        #1) NEURONAL RESPONSES FOR NEURONS neuron.N(maximum_location_in_degrees, activation_width, Amplitude=1):
        angle=self.N_angles
        #angle = ny.arange(0.0, 360, 360.0/self.N_number)
        self.neurons = [Neurons.N(self.width, self.N_number, self.N_angles).neuron_gauss(degree) for degree in angle]
        self.vec = ny.matrix([[ny.cos(fi*(ny.pi/180)), ny.sin(fi*(ny.pi/180))] for fi in angle])


        #2) NEURONAL ACTIVITY AT POINT i IN DEGREES E.G.: Neuron1.neuron_gauss(i)
        self.pop_background=[(self.neurons[i])[angle_outside] for i in ny.arange(0,len(angle))]
        self.pop_square=[(self.neurons[i])[angle_inside] for i in ny.arange(0,len(angle))]
        self.square_x=ny.arange((main_size/2)-(square_size/2),(main_size/2)+(square_size/2)+1)
        self.square_y=ny.arange((main_size/2)-(square_size/2),(main_size/2)+(square_size/2)+1)

    def neuronal_activity(self, angle):
        activity=[(self.neurons[i])[angle] for i in ny.arange(0,len(self.N_angles))]
        return activity


    def initial_pop_code(self):
        pop=ny.zeros((self.main_size, self.main_size, self.N_number))
        for x in ny.arange(0.0, self.main_size):
            for y in ny.arange(0.0, self.main_size):
                if x in self.square_x and y in self.square_y:
                    pop[x,y,:]=self.pop_square
                else:
                    pop[x,y,:]=self.pop_background

        return pop


    def show_vectors(self, population_code, time_frames, i, all=True):

        self.all=all
        fig = pp.gcf()
        fig.add_subplot(math.floor(time_frames/3.0)+1,3,i+1)
        pp.subplots_adjust(left=None, bottom=None, right=None, top=0.8, wspace=0.4, hspace=0.5)
        pp.locator_params(tight=True, nbins=5)
        h_v_edges = ny.zeros(2)
        for x in ny.arange(0.0,self.main_size):
            for y in ny.arange(0.0,self.main_size):

                    multiple=ny.multiply(population_code[x,y,:],ny.transpose(self.vec))
                    x1=ny.sum(multiple[0,:])
                    y1=ny.sum(multiple[1,:])

                    if self.all:
#                        ax=pp.gca()
#                        ax.add_patch(fap((x,y),((x+(self.square_size/4)*x1/ny.sqrt(x1**2+y1**2)),((y+(self.square_size/4)*y1/ny.sqrt(x1**2+y1**2)))), arrowstyle='->',linewidth=0.5,mutation_scale=10))
                        degree = self.pop_degree(population_code, x, y)
                        angle=degree*(180/ny.pi)
                        self.I.pix_wise(x,29-y,angle) # shows the stimulus
                        pp.xlabel('Pixel')
                        pp.ylabel('Pixel')
                        if i>0:
                            pp.title('After %i model cycles' % i)
                        else:
                            pp.title('Model input')


                    else:
                        r=ny.sqrt((x1**2)+(y1**2))
                        x3=ny.arcsin(y1/r)
                        if (x,y)==((self.main_size/2)+(self.square_size/2), self.main_size/2):
                            h_v_edges[0]=x3*180/ny.pi
                        elif (x,y)==(self.main_size/2, (self.main_size/2)+(self.square_size/2)):
                            h_v_edges[1]=x3*180/ny.pi
        return h_v_edges

    def pop_degree(self, population_code, x , y):

        multiple=ny.multiply(population_code[x,y,:],ny.transpose(self.vec))

        x1=ny.sum(multiple[0,:])
        y1=ny.sum(multiple[1,:])

        if x1<0:
            angle=(ny.arctan(y1/x1)+ny.pi)
        elif x1>0 and y1>=0:
            angle=ny.arctan(y1/x1)
        elif x1>0 and y1<0:
            angle=(ny.arctan(y1/x1)+(2*ny.pi))
        elif x1==0 and y1>0:
            angle=ny.pi/2.0
        else:
            angle=3*ny.pi/2
        return angle

    def plot_pop(self, population_code, time_frames, i):

        # get current axes object
        #frame1 = pp.gca()
        # get current figure
        fig = pp.gcf()
        frame1 = fig.add_subplot(math.floor(time_frames/3.0)+1,3,i+1, frameon=False)
        pp.subplots_adjust(left=None, bottom=None, right=None, top=0.77, wspace=None, hspace=0.4)
        im = pp.imread('colorspace_polar_z=0.0.jpg')
        im = ny.fliplr(ny.rot90(im, k=2))
        # draw it on the canvas
        pp.imshow(im,figure=frame1)
        # hide axes
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        frame1.set_axis_off()

        for x in ny.arange(0.0,self.main_size):
            for y in ny.arange(0.0,self.main_size):
                if not ny.any(population_code[x,y,:])==0:
                    angle = self.pop_degree(population_code, x, y)
                    y2=ny.arange(0,ny.max(ny.absolute(population_code[x,y,:]))+0.01,0.01) # vector length is maximum of all neurons per pixel! not value for angle direction... change to mean or so???
                    x2=angle+(0.0*y2)
                    frame2 = fig.add_subplot(math.floor(time_frames/3.0)+1,3,i+1, polar=True, axisbg='none', frameon=False)
                    frame2.plot(x2,y2,'k')
                    pp.title('After %d Model Cycles' %i)
                    frame2.axes.get_xaxis().set_visible(False)
                    frame2.axes.get_yaxis().set_visible(False)
        pp.axhline(xmax=1.0, linewidth=0.7, color='w', linestyle='--')
        y2=ny.arange(0,ny.max(ny.absolute(population_code[:,:,:]))*1.01,0.01)
        pp.plot(45*(ny.pi/180)+(0.0*y2),y2,'w', linestyle='--')
        pp.plot(90*(ny.pi/180)+(0.0*y2),y2,'w', linestyle='--')
        pp.plot(135*(ny.pi/180)+(0.0*y2),y2,'w', linestyle='--')
        pp.plot(225*(ny.pi/180)+(0.0*y2),y2,'w', linestyle='--')
        pp.plot(270*(ny.pi/180)+(0.0*y2),y2,'w', linestyle='--')
        pp.plot(315*(ny.pi/180)+(0.0*y2),y2,'w', linestyle='--')
        #pp.axvline(ymin=0.04, ymax=0.97,linewidth=0.7, color='grey', linestyle='--')

    def plot_cartesian_pop(self, population_code, time_frames, i):
        """
        PLOT POPULATION CODE FOR ALL PIXELS:
        """
        for x in ny.arange(0.0,self.main_size):
            for y in ny.arange(0.0,self.main_size):
                if not ny.any(population_code[x,y,:])==0:

                    x1=ny.arange(0,self.N_number)
                    y1=[population_code[x,y,k] for k in x1]
                    for a in x1:
                        if y1[a]<0:
                            if a in ny.arange(0,self.N_number/2.0):
                                y1[a+(self.N_number/2)]-=y1[a]
                                y1[a]=0
                            else:
                                y1[a-(self.N_number/2)]-=y1[a]
                                y1[a]=0

                    ax=pp.subplot(math.floor(time_frames/3.0)+1,3,i+1)
                    ax.plot(x1,y1)
                    ax.grid(True)
                    #ax.xaxis.set_ticks_position("top")
                    pp.title('Neuron number \n After %d Model Cycles' %i)
                    pp.ylabel('Neuronal activation')
                    #pp.ylim(0.01,1.01)

                    #pp.show()


    def plot_row(self, x, y, population_code, i, time_frames, x_all=True):

        self.all=x_all
        fig = pp.gcf()
        fig.add_subplot(math.floor(time_frames/3.0)+1,3,i+1)
        pp.grid(True)
        pp.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.5)

        #pp.locator_params(tight=True, nbins=5)
        if self.all:
            for x in ny.arange(0.0,self.main_size):
                angle =(180.0/ny.pi)*self.pop_degree(population_code, x, y)
                pp.scatter(x,angle)
                pp.ylim(0,360)
                pp.title('After %d Model Cycles' %i)
                pp.xlabel('all x pixels for y=%d' %y)
                pp.ylabel('degree')

        else:
            for y in ny.arange(0.0,self.main_size):
                angle = (180/ny.pi)*self.pop_degree(population_code, x, y)
                pp.scatter(y,angle)
                pp.title('After %d Model Cycles' %i)
                pp.xlabel('all y pixels for x=%d' %x)
                pp.ylabel('direction in color space in degree')

    def heat_map(self, population_code, i, time_frames, A):

        for x in ny.arange(0.0,self.main_size):
            for y in ny.arange(0.0,self.main_size):
                A[x,y,i] = self.pop_degree(population_code, x, y)

        if i>0:

            D=(A[:,:,i-1]*180.0/ny.pi)-(A[:,:,i]*180.0/ny.pi)
            fig = pp.gcf()
            frame1 = fig.add_subplot(math.floor(time_frames/3.0)+1,3,i)
            pp.imshow(D, interpolation='nearest', figure=frame1, vmin=-90, vmax=90)
            frame2 = fig.add_subplot(math.floor(time_frames/3.0),3,i)
            range=ny.arange((self.main_size/2)-(self.square_size/2), (self.main_size/2)+(self.square_size/2),0.001)
            start=(self.main_size/2)-(self.square_size/2)+(0*range)
            stop=(self.main_size/2)+(self.square_size/2)+(0*range)
            frame2.plot(range,start, 'w')
            frame2.plot(range,stop, 'w')
            frame2.plot(start,range, 'w')
            frame2.plot(stop,range, 'w')
            pp.xlim(0,29)
            pp.ylim(0,29)
            pp.title('Change in color-space-direction \n from model cycle %i to %i' %((i-1), i))
            pp.xlabel('pixel')
            pp.ylabel('pixel')
            pp.colorbar()

    def twoD_activation(self, population_code, c, i):

        fig = pp.gcf()
        pp.suptitle('Neuronal activations in %i. model cycle' %i)
        #frame1 = fig.add_subplot(6,self.N_number,(c*self.N_number)+1+k)
        grid = AxesGrid(fig, int(''.join(map(str,[6,1,c+1]))),
            nrows_ncols = (1, self.N_number),
            axes_pad = 0.0,
            share_all=True,
            #label_mode = "1",
            cbar_size="7%",
            cbar_mode="single",
        )
        range=ny.arange((self.main_size/2)-(self.square_size/2), (self.main_size/2)+(self.square_size/2),0.001)
        start=(self.main_size/2)-(self.square_size/2)+(0*range)
        stop=(self.main_size/2)+(self.square_size/2)+(0*range)
        for k in ny.arange(0,self.N_number):
            im=grid[k].imshow(population_code[:,:,k],origin="lower",  vmin=ny.min(population_code), vmax=ny.max(population_code), interpolation='nearest')
            grid[k].plot(range,start, 'w')
            grid[k].plot(range,stop, 'w')
            grid[k].plot(start,range, 'w')
            grid[k].plot(stop,range, 'w')
            grid[k].set_xticks(ny.arange(0,29,10))
            grid[k].set_yticks(ny.arange(0,29,10))




            if not k:

                if not c:
                    grid[k].axes.yaxis.set_label_position('left')
                    grid[k].axes.yaxis.set_label_text('After 1st \n stage of V1')
                    grid[k].axes.xaxis.set_label_position('top')
                    grid[k].axes.xaxis.set_label_text('Neuron %i \n (%.1f$^\circ$)' %(k,k*(360.0/self.N_number)))

                elif c==1:
                    grid[k].axes.yaxis.set_label_position('left')
                    grid[k].axes.yaxis.set_label_text('After 2nd \n stage of V1')
                elif c==2:
                    grid[k].axes.yaxis.set_label_position('left')
                    grid[k].axes.yaxis.set_label_text('After 3rd \n stage of V1')
                elif c==3:
                    grid[k].axes.yaxis.set_label_position('left')
                    grid[k].axes.yaxis.set_label_text('After 1st \n stage of MT')
                elif c==4:
                    grid[k].axes.yaxis.set_label_position('left')
                    grid[k].axes.yaxis.set_label_text('After 2nd \n stage of MT')
                else:
                    grid[k].axes.yaxis.set_label_position('left')
                    grid[k].axes.yaxis.set_label_text('After 3rd \n stage of MT')
                    grid[k].axes.xaxis.set_label_position('top')
                    grid[k].axes.xaxis.set_label_text('pixel')


            if not c and k:
                grid[k].axes.xaxis.set_label_position('top')
                grid[k].axes.xaxis.set_label_text('Neuron %i \n (%.1f$^\circ$)' %(k,k*(360.0/self.N_number)))
            if c==5:
                grid[k].axes.xaxis.set_label_position('bottom')
                grid[k].axes.xaxis.set_label_text('pixel')



            grid.cbar_axes[0].colorbar(im)






if __name__ == '__main__':
    angles=ny.array((0,20,40,60,90,120,140,160,180,200,220,240,270,300,320,340))
    p = Population(30, 4, 30, 16, angles, 45, 90)
    print 30, 4, 30, 16, angles, 45, 90
    pop=p.initial_pop_code()
    #p.show_vectors(pop,0,0)
    p.plot_row(15,1,pop,1,1, x_all=False)
    #p.twoD_activation(pop,5,1)

    pp.show()





