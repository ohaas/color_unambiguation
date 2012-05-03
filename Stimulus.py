__author__ = 'ohaas'

import Image, ImageDraw
import matplotlib.pyplot as pp
from matplotlib.patches import FancyBboxPatch as fbp


class image(object):

    def __init__(self,main_size,square_size):
        self.i=Image.new("RGB",(main_size,main_size),color=(0,0,200))
        draw = ImageDraw.Draw(self.i)
        draw.rectangle([(((main_size/2)-(square_size/2)), (main_size/2)-(square_size/2)-1), ((main_size/2)+(square_size/2), (main_size/2)+(square_size/2)-1)],fill=(200,0,0))
        pp.imshow(self.i, interpolation="nearest")

image(30,10)
pp.show()