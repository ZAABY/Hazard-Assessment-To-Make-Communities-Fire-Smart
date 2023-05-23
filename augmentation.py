
import os
from osgeo import gdal, osr
import numpy as np
import slidingwindow as sw
import xml.etree.ElementTree as ET
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
import glob
from pascal_voc_writer import Writer
import math
import random
import matplotlib.pyplot as plt
import torch
import random
import imgaug
import imgaug.augmenters as iaa 
from imgaug.augmentables.bbs import BoundingBoxesOnImage, BoundingBox


def update_root_with_bbox(new_bboxes, root, name_ext):
    old_fname = root.find("filename").text
    new_fname = old_fname[:-4] + "_" + str(name_ext) + old_fname[-4:]
    # print(old_fname, new_fname)
    root.find("filename").text = new_fname

    for i, obj in enumerate(root.findall("object")):

        classname = obj.find("name").text

        
        x1 = new_bboxes[i].x1
        x2 = new_bboxes[i].x2
        y1 = new_bboxes[i].y1
        y2 = new_bboxes[i].y2

        

        obj.find("bndbox").find("xmin").text = str(int(x1))
        obj.find("bndbox").find("xmax").text = str(int(x2))
        obj.find("bndbox").find("ymin").text = str(int(y1))
        obj.find("bndbox").find("ymax").text = str(int(y2))



#TODO
def rand_rotate(image, bboxes, root, name_ext):
    """

    Arguments:
        image (PIL.Image):
        bboxes (BoundingBoxesOnImage): 
        root (ET.root): 
        name_ext (int): modification to the filename
    """

    seq = iaa.Sequential([
        iaa.Rotate((-30, 30))
    ])

    new_image, new_bboxes = seq(image=np.array(image), 
                                bounding_boxes=bboxes)

    update_root_with_bbox(new_bboxes, root, name_ext)

 
    return Image.fromarray(new_image), root


#TODO
def rand_flip(image, bboxes, root, name_ext):
    fliplr = iaa.Fliplr(1)
    flipud = iaa.Flipud(0.5)
    new_image, new_bboxes = fliplr(image=np.array(image), 
                                bounding_boxes=bboxes)
    new_image, new_bboxes = flipud(image=new_image, 
                                bounding_boxes=new_bboxes)
    update_root_with_bbox(new_bboxes, root, name_ext)
    return Image.fromarray(new_image), root


#TODO
def rand_shear(image, bboxes, root, name_ext):
    shearx = iaa.ShearX((-20, 20))
    sheary = iaa.ShearY((-20, 20))
    new_image, new_bboxes = shearx(image=np.array(image), 
                                bounding_boxes=bboxes)
    new_image, new_bboxes = sheary(image=new_image, 
                                bounding_boxes=new_bboxes)
    update_root_with_bbox(new_bboxes, root, name_ext)
    return Image.fromarray(new_image), root


#TODO
def rand_brightness(image, bboxes, root, name_ext):
    brig = iaa.AddToBrightness((-30, 30))
    contrast = iaa.LogContrast((0.6, 1.4))
    new_image, new_bboxes = brig(image=np.array(image), 
                                bounding_boxes=bboxes)
    new_image, new_bboxes = contrast(image=new_image, 
                                bounding_boxes=new_bboxes)
    update_root_with_bbox(new_bboxes, root, name_ext)
    return Image.fromarray(new_image), root


def rand_colours(image, bboxes, root, name_ext):
    """ Applies channel-wise augmentation by adding a bias to
    each channel of the image. 

    for c in image channels:
        image[c] += random_value

    @Params:
        image (np.ndarray[C, W, H]): An image array in channel-first format
        bboxes (BoundingBoxesOnImage): All boxes on an image
        root (ET.ElementTree.Element): The root of the original XML tree
        name_ext (int): Index of the annotation. This will be appended to the 
                        filename of the image and annotation file, before the file extension.

    @Returns:
        new_image (np.ndarray[C, H, W]): The augmented image
        root (ET.ElementTree.Element): The updated root
    
    """

    new_image = np.zeros(image.shape)

    for c in range(image.shape[0]):
        new_image[c] = np.minimum(np.maximum(image[c] + (0.7*np.random.rand() - 0.35)*np.mean(image[c]), 0), 1)
    update_root_with_bbox(bboxes, root, name_ext)
    return new_image, root


def rand_contrast(image, bboxes, root, name_ext):


    new_image = np.zeros(image.shape)

    new_image = np.minimum(np.maximum(((np.random.rand(image.shape[0], 1, 1) + 0.5) * image)  + 
                    (np.random.rand(image.shape[0], 1, 1) * 0.2 - 0.05) * np.mean(image, axis=0, keepdims=True), 0), 1)
    update_root_with_bbox(bboxes, root, name_ext)
    return new_image, root

        
#TODO

def rand_scale(image, bboxes, root, name_ext):
    scalex = iaa.geometric.ScaleX((2, 3))
    scaley = iaa.geometric.ScaleY((2, 3))
    new_image, new_bboxes = scalex(image=np.array(image), 
                                bounding_boxes=bboxes)
    new_image, new_bboxes = scaley(image=new_image, 
                                bounding_boxes=new_bboxes)
    update_root_with_bbox(new_bboxes, root, name_ext)
    return Image.fromarray(new_image), root

#TODO
def rand_all(image, bboxes, root, name_ext):
    brig = iaa.AddToBrightness((-30, 30))
    contrast = iaa.LogContrast((0.6, 1.4))
    scalex = iaa.geometric.ScaleX((1, 3))
    scaley = iaa.geometric.ScaleY((1, 3))
    fliplr = iaa.Fliplr(0.5)
    flipud = iaa.Flipud(0.5)

    seq = iaa.Sequential([
        brig,
        contrast,
        scalex,
        scaley,
        fliplr,
        flipud
    ])

    colour = iaa.Sequential([
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(0, iaa.Add((0, 20))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
    ])

    new_image, new_bboxes = seq(image=np.array(image), 
                                bounding_boxes=bboxes)
    # new_image, new_bboxes = brig(image=new_image, 
    #                             bounding_boxes=new_bboxes)
    # new_image, new_bboxes = contrast(image=new_image, 
    #                             bounding_boxes=new_bboxes)                            
    # new_image, new_bboxes = scalex(image=new_image, 
    #                             bounding_boxes=new_bboxes)
    # new_image, new_bboxes = scaley(image=new_image, 
    #                             bounding_boxes=new_bboxes)
    # new_image, new_bboxes = fliplr(image=new_image, 
    #                             bounding_boxes=new_bboxes)
    # new_image, new_bboxes = flipud(image=new_image, 
    #                             bounding_boxes=new_bboxes)
    # new_image, new_bboxes = contrast(image=new_image, 
    #                             bounding_boxes=new_bboxes)
    new_image, new_bboxes = colour(image=new_image, 
                                bounding_boxes=new_bboxes)
    update_root_with_bbox(new_bboxes, root, name_ext)
    return Image.fromarray(new_image), root






def h_flip(image, bboxes, root, name_ext):
    """

    image (np.ndarray[C, W, H])

    """
    fliplr = iaa.Fliplr(1)
    new_image = np.zeros(image.shape)


    for c in range(image.shape[0]):
        new_channel, new_bboxes = fliplr(image=np.moveaxis(image[c:c+1, :, :], 0, -1), 
                                bounding_boxes=bboxes)
        new_image[c] = np.moveaxis(new_channel, -1, 0)
    update_root_with_bbox(new_bboxes, root, name_ext)
    return new_image, root





def v_flip(image, bboxes, root, name_ext):
    """

    image (np.ndarray[C, W, H])

    """
    flipud = iaa.Flipud(1)
    new_image = np.zeros(image.shape)

    for c in range(image.shape[0]):
        new_channel, new_bboxes = flipud(image=np.moveaxis(image[c:c+1, :, :], 0, -1), 
                                    bounding_boxes=bboxes)
        new_image[c] = np.moveaxis(new_channel, -1, 0)
    update_root_with_bbox(new_bboxes, root, name_ext)
    return new_image, root


#TODO
def hv_flip(image, bboxes, root, name_ext):
    fliplr = iaa.Fliplr(1)
    flipud = iaa.Flipud(1)
    new_image = np.zeros(image.shape)

    for c in range(image.shape[0]):
        new_channel, new_bboxes = flipud(image=np.moveaxis(image[c:c+1, :, :], 0, -1), 
                                    bounding_boxes=bboxes)
        new_channel, new_bboxes = fliplr(image=new_channel, bounding_boxes=new_bboxes)
        new_image[c] = np.moveaxis(new_channel, -1, 0)
    update_root_with_bbox(new_bboxes, root, name_ext)
    return new_image, root
