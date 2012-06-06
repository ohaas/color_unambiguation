__author__ = 'ohaas'

import Image, ImageDraw
import matplotlib.pyplot as pp
from matplotlib.patches import FancyBboxPatch as fbp
import numpy as ny


class image(object):

    def __init__(self,main_size,square_size, angle_outside , angle_inside):
        C = ny.genfromtxt("Colormatrix.txt")
        t_out=ny.round((angle_outside*(2*ny.pi/360))/(1/60.0)).astype(int)
        t_in=ny.round((angle_inside*(2*ny.pi/360))/(1/60.0)).astype(int)
        (r_out, g_out, b_out)=C[t_out,:]
        (self.r, self.g, self.b)=C[t_in,:]
        print r_out, g_out, b_out
        print self.r, self.g, self.b
        self.main_size=main_size
        self.square_size=square_size
        self.i=Image.new("RGB",(main_size+1,main_size+1),color=(ny.round(r_out*255).astype(int),ny.round(g_out*255).astype(int),ny.round(b_out*255).astype(int)))

    def pic(self):
        draw = ImageDraw.Draw(self.i)
        draw.rectangle([(((self.main_size/2)-(self.square_size/2)), (self.main_size/2)-(self.square_size/2)),
            ((self.main_size/2)+(self.square_size/2), (self.main_size/2)+(self.square_size/2))],fill=(ny.round(self.r*255).astype(int),ny.round(self.g*255).astype(int),ny.round(self.b*255).astype(int)))
        pp.imshow(self.i, interpolation="nearest")


if __name__ == '__main__':
    I=image(30,10, 135, 180)
    I.pic()
    pp.show()
    # 0.32243642,  0.67756358,  0.85197259  in: 0.25000442,  0.74999558,  0.49702585
