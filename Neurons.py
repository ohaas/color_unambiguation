__author__ = 'ohaas'

import numpy as ny
import matplotlib.pyplot as pp
import colorspace


def gauss(x, mu, sigma):
    """
        mu IS WHERE THE MAXIMUM IS LOCATED, SIGMA SQUARED IS THE WIDTH AND A IS THE AMPLITUDE
        """
    return ny.exp(-(x-mu)**2/(2.0*sigma**2))



class N(object):


    def __init__(self, sigma, N_number,N_angles, A=1):
        self.sigma=sigma
        self.A=A
        self.N_number=N_number
        self.N_angles=N_angles

    def neuron_gauss(self, mu):
        y=ny.zeros(361.0)
        x2=ny.arange( 0.0, 361.0, 1)
        for x in x2:
            y[x]= self.A*(gauss(x, mu, self.sigma) + self.A*gauss(x, mu-360, self.sigma) + self.A*gauss(x, mu+360, self.sigma))
        return y

    def plot_act(self):
        angle=self.N_angles
        #angle = ny.arange(0.0, 360, 360.0/self.N_number)
        neurons = [N(self.sigma, self.N_number, self.N_angles).neuron_gauss(degree) for degree in angle]
        x3=ny.arange(0,2*ny.pi,2*ny.pi/361)
        ax = pp.subplot(111,polar=True)
        for i in ny.arange(0,len(angle)):
            ax.plot(x3,neurons[i], label='$neuron$ $%i$ $(\mu=%.2f\degree)$' %(i+1,angle[i]))#i*(360.0/self.N_number)))
        pp.xlim(0,360)
        y=ny.arange(0,1.01,0.01)
        x=0.0*y
        pp.ylim(0,1)
        # Shink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])


        pp.polar(x,y,color='blue', ls='--', lw=3, label='$Amplitude$\n$of$ $neuronal$\n$activation$')

        pp.xlabel('Spacial orientation in degree with neuron %s=%.1f' % (u"\u03C3",self.sigma))
        pp.title('Neuronal tuning curves')

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))



if __name__ == '__main__':
    angles=ny.array((0,20,40,60,90,120,140,160,180,200,220,240,270,300,320,340))
    #angles=ny.arange(0,360,360.0/16)
    N(50.0,8, angles).plot_act()
    pp.show()