__author__ = 'ohaas'

import numpy as ny
import matplotlib.pyplot as pp

def gauss(x, mu, sigma):
    """
        mu IS WHERE THE MAXIMUM IS LOCATED, SIGMA SQUARED IS THE WIDTH AND A IS THE AMPLITUDE
        """
    return ny.exp(-(x-mu)**2/(2.0*sigma**2))

class N(object):


    def __init__(self, sigma, A=1):
        self.sigma=sigma
        self.A=A


    def neuron_gauss(self, mu):
        y=ny.zeros(361.0)
        x2=ny.arange( 0.0, 361.0, 1)
        for x in x2:
            y[x]= self.A*(gauss(x, mu, self.sigma) + self.A*gauss(x, mu-360, self.sigma) + self.A*gauss(x, mu+360, self.sigma))
        return y