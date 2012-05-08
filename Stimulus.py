__author__ = 'ohaas'

import Image, ImageDraw
import matplotlib.pyplot as pp
from matplotlib.patches import FancyBboxPatch as fbp


class image(object):

    def __init__(self,main_size,square_size):
        self.main_size=main_size
        self.square_size=square_size
        self.i=Image.new("RGB",(main_size+1,main_size+1),color=(0,0,200))

    def pic(self):
        draw = ImageDraw.Draw(self.i)
        draw.rectangle([(((self.main_size/2)-(self.square_size/2)), (self.main_size/2)-(self.square_size/2)-1),
            ((self.main_size/2)+(self.square_size/2), (self.main_size/2)+(self.square_size/2)-1)],fill=(200,0,0))
        pp.imshow(self.i, interpolation="nearest")


if __name__ == '__main__':
    image(30,10)
    pp.show()

