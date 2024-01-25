#! /usr/bin/env python
""" Various Utilities
"""
from __future__ import print_function

from OpenGL.GL import *
from csgl import *

import glfw
import math as mathf

from PIL import Image
from PIL.Image import open as pil_open
import imageio
import glob

def screenshot(width,height, file_name_out=None):
    data = glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,outputType=None)
    image = Image.frombytes(mode="RGB", size=(width, height), data=data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(file_name_out)
    return image

def gen_video(frames_in_path, video_out_path, mp4save_fn):
    writer = imageio.get_writer(f'{video_out_path}/{mp4save_fn}', mode='I', fps=30, codec='libx264', bitrate='16M')
    frame_names = sorted(glob.glob(frames_in_path+'*.png'))
    for fn in frame_names:
        f = imageio.imread(fn)
        writer.append_data(f)
    writer.close()
