__author__ = 'ohaas'

import numpy as ny
import matplotlib.pyplot as pp
#
#x=3
#try:
#    x
#except NameError:
#    pass
#else:
#    print 'yes'
x=ny.arange(0,10,0.01)
y=ny.exp(-x/2.0)+0.08
print ny.exp(-5/2.0)
print ny.exp(-5/2.0)+0.08
pp.plot(x,y)
pp.show()