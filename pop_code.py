__author__ = 'ohaas'

import numpy as ny
import matplotlib.pyplot as pp
from matplotlib.patches import FancyArrowPatch as fap
import Stimulus
import Neurons

class Population(object):


    def __init__(self, main_size, square_size, gauss_width):
        """
        main_size= IMAGE SIZE SQUARED (main_size x main_size), square_size=STIMULUS SIZE SQUARED,
        start=STIMULUS STARTING POINT SQUARED (lower left stimulus corner), gauss_width= WIDTH OF NEURONAL GAUSS TUNING CURVE
        """
        self.main_size = main_size
        self.square_size = square_size
        self.I = Stimulus.image(main_size,square_size)
        self.width=gauss_width

        self.vec = ny.matrix(((0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)))

        #1) NEURONAL RESPONSES FOR NEURONS neuron.N(maximum_location_in_degrees, activation_width, Amplitude=1):
        angle = ny.arange(0.0, 360, 45.0)
        neurons = [Neurons.N(self.width).neuron_gauss(degree) for degree in angle]

        #2) NEURONAL ACTIVITY AT POINT X IN DEGREES E.G.: Neuron1.neuron_gauss(X)
        self.pop_background=[(neurons[i])[45.0] for i in ny.arange(0,len(angle))]
        self.pop_square=[(neurons[i])[90.0] for i in ny.arange(0,len(angle))]
        self.square_x=ny.arange((main_size/2)-(square_size/2),(main_size/2)+(square_size/2)+1)
        self.square_y=ny.arange((main_size/2)-(square_size/2)-1,(main_size/2)+(square_size/2))


    def initial_pop_code(self):
        pop=ny.zeros((self.main_size, self.main_size, 8))
        for x in ny.arange(0.0, self.main_size):
            for y in ny.arange(0.0, self.main_size):
                if x in self.square_x and y in self.square_y:
                    pop[x,y,:]=self.pop_square
                else:
                    pop[x,y,:]=self.pop_background
        return pop


    def show_vectors(self, population_code, all=True):
        self.all=all
        h_v_edges = ny.zeros(2)
        for x in ny.arange(0.0,self.main_size):
            for y in ny.arange(0.0,self.main_size):

                    multiple=ny.multiply(population_code[x,y,:],ny.transpose(self.vec))
                    x1=ny.sum(multiple[0,:])
                    y1=ny.sum(multiple[1,:])

                    if self.all:
                        ax=pp.gca()
                        pp.axis([0,self.main_size,0, self.main_size])
                        ax.add_patch(fap((x,y),((x+(self.square_size/4)*x1/ny.sqrt(x1**2+y1**2)),((y+(self.square_size/4)*y1/ny.sqrt(x1**2+y1**2)))), arrowstyle='->',linewidth=0.5,mutation_scale=10))
                        self.I.pic() # shows the stimulus


                    else:
                        r=ny.sqrt((x1**2)+(y1**2))
                        x3=ny.arcsin(y1/r)
                        if (x,y)==((self.main_size/2)+(self.square_size/2), self.main_size/2):
                            h_v_edges[0]=x3*180/ny.pi
                        elif (x,y)==(self.main_size/2, (self.main_size/2)+(self.square_size/2)):
                            h_v_edges[1]=x3*180/ny.pi
        return h_v_edges

    def plot_pop(self, population_code,t):
        """
        PLOT POPULATION CODE FOR ALL PIXELS:
        """
        for a in ny.arange(0,self.main_size):
            for b in ny.arange(0,self.main_size):
                multiple=ny.multiply(population_code[a,b,:],ny.transpose(self.vec))
                x=ny.sum(multiple[0,:])
                y=ny.sum(multiple[1,:])
                # connect (0,0) with (x,y):
                r=ny.sqrt((x**2)+(y**2))
                angle=(ny.arcsin(y/r))
                y1=ny.arange(0,1.01,0.01)
                x1=angle+(0.0*y1)
                pp.polar(x1,y1)
        pp.xlabel('Neuron number')
        pp.ylabel('Neuronal activation')
        pp.suptitle('Population code after %d model cycles' %t)
        pp.show()




if __name__ == '__main__':
    p = Population(30, 6, 2, 30)
    pop=p.initial_pop_code()
    pop.show_vectors()
    pp.show()





