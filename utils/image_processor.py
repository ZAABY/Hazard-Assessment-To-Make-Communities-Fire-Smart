"""
image_processor.py

Handles reading and writing GeoTiffs, tiling,
resizing, PCA reduction, preprocessing, and 
much, much more. General purpose image processing 
library.

Author: Zony Yu

"""




import os
import sys

import torch
from tqdm import tqdm
from osgeo import gdal, osr
import numpy as np
from pascal_voc_writer import Writer
import slidingwindow as sw
import glob
from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt
from copy import deepcopy
import xml.etree.ElementTree as ET
from imgaug import BoundingBox, BoundingBoxesOnImage
from pathlib import Path



currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)





Image.MAX_IMAGE_PIXELS = None
GDAL_DATA_TYPE = gdal.GDT_Byte
GEOTIFF_DRIVER_NAME = r'GTiff'
NO_DATA = -1500
SPATIAL_REFERENCE_SYSTEM_WKID = 4326
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")





class Image_processor:
    def __init__(self):
        self.windows = None
    
    def __call__(self, img_path_in, img_path_out, label_path_in=None, label_path_out=None, tile_size=128, overlap=0.1):
        """ This function takes all images and labels in the input paths,
        tiles them, and saves it to the output paths. If label_path_in and 
        label_path_out are not specified, this function will only tile the images
        
        ! NOTE !
        Due to the fact that XML annotations can often be used with multiple images (i.e.
        we can have one annotation file and use it with 3-channel, or n-channel images of the same
        crop), it is impractical to read the image <filename> from the XML file. Therefore, this
        tiling method is the ONLY method that matches the images to its annotation via the filename only.
        For example, 
            IMG_456.tif

        will prompt the program to search for:
            IMG_456.xml

        ! THE IMAGE AND ANNOTATION MUST HAVE THE SAME FILENAME. !

        All other methods handling labels and images will search for the image
        based on the <filename> tag inside the XML file. This is fine, because 
        this method will ensure that all tiled XML files will have the <filename>
        attribute associated with the correct image.

        In other words, it is MANDATORY to run this function if you want to use self.preprocess(),
        or Generator(), as this function syncs all XML files' <filename> attributes.

        @Params
            img_path_in (string): path to the folder containing input images to be tiled
            img_path_out (string): path to the folder to store the tiles
            label_path_in (string): path to the folder containing labels 
            label_path_out (string): path to put the tiled labels
            tile_size (int): size of square tiles
        """
        image_files = self.all_img_extensions(img_path_in)
        
        print(image_files)
        
        for img_path in image_files:
            
            # Pull the filenames out of the path
            # img_filename: some_image.png
            # filename: some_image
            # label_filename: some_image.xml
            img_filename = img_path.split("/")[-1]    
            filename = "".join(img_filename.split(".")[:-1])
            label_filename = "".join(img_filename.split(".")[:-1]) + ".xml"
            
            # Read Image
            img_array = self.read_image(img_path)
            c, h, w = img_array.shape
            print(f"Processing {img_path}...")
            
            # if label_path_in is not specified, 
            # then do not read XML
            labels = None
            if label_path_in is not None:
                labels = self.read_xml(os.path.join(label_path_in, label_filename))
            
            # self.tile_image also saves an instance of windows as self.windows
            # labels are stored in each window
            tile_arrays, windows = self.tile_image(img_array, tile_size=tile_size, labels=labels, overlap=overlap)
            for i in tqdm(range(len(windows)), desc="Writing tile:"):
                tile_ext = "_TILE_" + str(i)
                for label in windows[i].labels:
                    label["filename"] += tile_ext
                
                # write the image tiles out
                # saves to png if channels <= 3, otherwise tiff.
                
                tile_path = ""
                if c <= 3:
                    tile_path = os.path.join(img_path_out, filename + tile_ext + ".png")
                    self.write_image(tile_arrays[i], tile_path)
                else:
                    tile_path = os.path.join(img_path_out, filename + tile_ext + ".tif")
                    self.write_image(tile_arrays[i], tile_path)
                
                # if label_path_out is not specified, do not
                # attempt to save labels
                if label_path_out is not None:
                    local_label_path = os.path.join(label_path_out, filename + tile_ext + ".xml")
                    self.write_xml(windows[i].labels, tile_path, local_label_path, windows[i].w, windows[i].h, c)
    

    
    def all_img_extensions(self, img_path_in):
        """ Returns a list of all image paths within a folder
        with the extensions png, jpg, tiff, and all the various 
        forms of those.
        
        @Params:
            img_path_in (string): Path to the folder containing the images
            
        @Returns:
            image_files ([string]): List of all image paths with the 
                                    extentions shown below
        
        """
        image_files = glob.glob(os.path.join(img_path_in, "*.png"))
        image_files.extend(glob.glob(os.path.join(img_path_in, "*.PNG")))
        image_files.extend(glob.glob(os.path.join(img_path_in, "*.jpg")))
        image_files.extend(glob.glob(os.path.join(img_path_in, "*.JPEG")))
        image_files.extend(glob.glob(os.path.join(img_path_in, "*.jpeg")))
        image_files.extend(glob.glob(os.path.join(img_path_in, "*.JPG")))
        image_files.extend(glob.glob(os.path.join(img_path_in, "*.tif")))
        image_files.extend(glob.glob(os.path.join(img_path_in, "*.tiff")))
        image_files.extend(glob.glob(os.path.join(img_path_in, "*.TIFF")))
        image_files.extend(glob.glob(os.path.join(img_path_in, "*.TIF")))
        
        return image_files
    
    def create_raster(self, 
                      output_path,
                      columns,
                      rows,
                      nband = 1,
                      gdal_data_type = GDAL_DATA_TYPE ,
                      driver = GEOTIFF_DRIVER_NAME):
        ''' Returns gdal data source raster object.
        Used by self.numpy_array_to_raster.
        
        @Params:
            output_path (string): Path to save the raster
            columns (int): width of the image in pixels
            rows (int): height of the image in pixels
            nband (int): number of bands (channels)
            gdal_data_type: datatype to save the raster
            driver: driver used to create the raster
        
        @Returns:
            output_raster (Dataset): The output GDAL Dataset
        '''
        # create driver
        driver = gdal.GetDriverByName(driver)

        output_raster = driver.Create(output_path,
                                    int(columns),
                                    int(rows),
                                    nband,
                                    eType = gdal_data_type)    
        return output_raster


    def crop_tif(self, tif_file, crop_bounds):
        ''' Crops a tif file to the crop_bounds specified
        Preserve the geospatial projection of the tif (crop is also georeferenced)
        Image is saved to same path as tif_file with '_crop.tif' appended to filename
        
        @Params:
            tif_file (str): the path to the tif file to crop
            crop_bounds (list): [xmin, ymin, width, height] of the rectangular crop bounds
            
        @Returns:
            out_path: the path of the crop that was saved
            output array: the cropped image as an array
        '''
        # Read the tif file
        ortho = gdal.Open(tif_file)
        # Crop ortho
        out_path = tif_file.replace('.tif', '_crop.tif')
        ortho = gdal.Translate(out_path, ortho, srcWin = crop_bounds)
        return out_path, np.moveaxis(ortho.ReadAsArray(), 0, -1)
    
    def draw_boxes(self, img_path, annot_path):
        """ Draws boxes on an image. Annotations must be 
        in the PASCAL VOC format
        
        @Params:
            img_path (string): path to an image file. Image can be 
                                either jpeg, png, or tiff
            annot_path (string): path to a label xml file.
                                File must be in the PASCAL VOC
                                format
                                
        @Returns:
            im_array (np.ndarray[C, H, W]): Numpy array in the channel
                                first format.
        """
        im_array = self.read_image(img_path)
        labels = self.read_xml(annot_path)
        
        for m in tqdm(range(len(labels)), desc = "Number of Boxes"):
            x1 = min(max(labels[m]["x1"]-1, 0), im_array.shape[2]-1)
            y1 = min(max(labels[m]["y1"]-1, 0), im_array.shape[1]-1)
            x2 = min(max(labels[m]["x2"]-1, 0), im_array.shape[2]-1)
            y2 = min(max(labels[m]["y2"]-1, 0), im_array.shape[1]-1)
            
            for i in range(im_array.shape[0]):
                max_val_at_channel = 1.
                im_array[i, y1:y2, x1] = max_val_at_channel
                im_array[i, y1:y2, x2] = max_val_at_channel
                im_array[i, y1, x1:x2] = max_val_at_channel
                im_array[i, y2, x1:x2] = max_val_at_channel
                
        return im_array
                
    
    def draw_boxes_with_class(self, img_path, label_path, show_plot=True, save_img=False):
        """
        Draws bounding boxes on image. Bounding boxes will be 
        colour-coded based on what class the predictions are.

        @Params:
            img_path (str): path to the image where the boxes are 
                            to be drawn on.
            label_path (str): path to the VOC XML labels.
            show_plot (bool): display the boxed image if true
            save_img (bool): save image if true.
        """
        from preprocessing import name_to_idx, idx_to_name

        random.seed(40)
        colours = Color()

        img = self.read_image(img_path)
        labels = self.read_xml(label_path)
        assert img.shape[0] == 3, "Only accepts 3 channel images"

        im = Image.fromarray((np.moveaxis(img, 0, -1)*255).astype(np.uint8))
        draw = ImageDraw.Draw(im)

        con_count = 0
        dec_count = 0
        snag_count = 0

        for obj in labels:
            x1 = obj["x1"]
            y1 = obj["y1"]
            x2 = obj["x2"]
            y2 = obj["y2"]

            name = obj["class"]

            if name == 'con':
                con_count +=1
            elif name == 'dec':
                dec_count +=1
            else:
                snag_count +=1

            name_idx = name_to_idx[name]
            conf = float(obj["conf"])

            draw.rectangle([x1, y1, x2, y2], outline=colours[name_idx], width=5)
            
            try:
                font = ImageFont.truetype(font="Ubuntu-C.ttf", size=15)
            except:
                try: 
                    font = ImageFont.truetype(font="arial.ttf", size=15)
                except:
                    font = ImageFont.load_default()


            draw.text((x1, y1), f"conf = {conf:.4f} and class = {name}", (255, 255, 255), font=font)

        if show_plot:
            plt.rcParams['figure.figsize'] = (10, 10)
            plt.imshow(im)
            plt.show()
        if save_img:
            im = np.moveaxis(np.array(im), -1, 0)/255.0
            path = str(Path("predictions") / img_path.split("/")[-1])
            self.write_image(im, path)
            print(f"Image saved to {path}")

        print(f'Total con trees: {con_count}')
        print(f'Total dec trees: {dec_count}')
        print(f'Total snag trees: {snag_count}')
        




    
    def find_local_labels(self, x, y, w, h, labels):
        """ Finds all the local bounding boxes given
        the sliding window coordinates. Also performs
        coordinate transformation to the bounding box 
        positions
        
        @Params:
            x (int): x-coordinate of the top-left corner of the 
                    sliding window
            y (int): y-coordinate of the top-left corner of the 
                    sliding window
            w (int): width of the sliding window
            h (int): height of the sliding window
            labels ([dict]): A list of dictionaries, where each 
                    dictionary represents a bounding box prediction
                    on the image. Each dictionary has these keys:
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "filename",
                        "class",
                        "conf"
        """
        local_labels = []
        for label in labels:
            if label["x1"] >= x and label["x2"]  <= x + w:
                if label["y1"] >= y and label["y2"] <= y + h:
                    new_label = deepcopy(label)
                    new_label["x1"] = label["x1"] - x
                    new_label["x2"] = label["x2"] - x
                    new_label["y1"] = label["y1"] - y
                    new_label["y2"] = label["y2"] - y
                    local_labels.append(new_label)
                    
        return local_labels
            
            
    
    def get_tiles(self, im_array, tile_size, overlap):
        ''' Uses a python package (slidingwindow) to generate windows over the orthophoto. 

        @Params: 
            im_array (np.ndarray), the orthomosaic for which predictions will be generated
            tile_size (int): the size of the tiles in px 
            overlap (float): the % amount the tiles will overlap

        @Returns:
            windows: a list of windows to be used to slice the image
        '''
        
        
        # Returns a list of windows, each window as (self.x, self.y, self.w, self.h)
        windows = sw.generate(im_array, sw.DimOrder.ChannelHeightWidth, tile_size, overlap)
        
        for window in windows:
            window.labels = []
            
        
        return windows
    
    
    def numpy_array_to_raster(self,
                              output_path,
                              numpy_array,
                              geotransform,
                              projection=None,
                              no_data = NO_DATA,
                              spatial_reference_system_wkid = SPATIAL_REFERENCE_SYSTEM_WKID,
                              driver = GEOTIFF_DRIVER_NAME):
        ''' Converts Numpy Array to GDAL raster

        @Params:
            output_path (string): full path to the raster to be written to disk
            numpy_array (np.ndarray[C, H, W]): numpy array containing data to write to raster
            no_data (int): value in numpy array that should be treated as no data
            spatial_reference_system_wkid: well known id (wkid) of the spatial reference of the data
            driver: string value of the gdal driver to use
            
        @Returns:
            output_raster (Dataset): A GDAL Dataset object converted from the 
                                    numpy array.

        '''
        nband, rows, columns = numpy_array.shape
        
        if nband > 3:
            gdal_data_type = gdal.GDT_UInt16
            numpy_array = (numpy_array * 2**16).astype(np.uint16)
        else:
            gdal_data_type = gdal.GDT_Byte
            numpy_array = (numpy_array * 255).astype(np.uint8)

        # create output raster
        output_raster = self.create_raster(output_path,
                                        int(columns),
                                        int(rows),
                                        nband,
                                        gdal_data_type) 
        
        
        for i in range(nband):
            if projection is None:
                spatial_reference = osr.SpatialReference()
                spatial_reference.ImportFromEPSG(spatial_reference_system_wkid)
                output_raster.SetProjection(spatial_reference.ExportToWkt())
            else:
                output_raster.SetProjection(projection)
                
            if geotransform is None:
                output_raster.SetGeoTransform((0, 1, 0, 0, 0, -1))
            else:
                output_raster.SetGeoTransform(geotransform)
            output_band = output_raster.GetRasterBand(i+1)
            output_band.WriteArray(numpy_array[i])         
            output_band.SetNoDataValue(no_data)
            output_band.FlushCache()
    

        if os.path.exists(output_path) == False:
            raise Exception('Failed to create raster: %s' % output_path)

        return output_raster
    
    def pca(self, img, output_channels=3):
        """
        Computes the PCA matrix for a single image, 
        and uses the matrix to reduce the image.
        Returns a numpy array raster with channels
        == output_channels

        @Params
            img (np.ndarray[C, H, W]) 
            output_channels (int): the desired number of output
                                channels
        @Returns:
            out (np.ndarray[C', H, W]): C' is the output_channels
        """
        x = torch.from_numpy(img)
        x = x.view(img.shape[0], -1)
        u = torch.mean(x, dim=-1, keepdim=True)
        s = torch.std(x, dim=-1, keepdim=True)
        z = (x-u)/(s + 1e-6)
        cov = torch.matmul(z, torch.transpose(z, 0, 1)) / x.shape[-1]
        
        # print(cov.shape)
        U, S, Vh = torch.svd(cov)
        # print(U.shape, S.shape, Vh.shape)
        
        M = torch.transpose(U[:, :output_channels], 0, 1)
        reduced = torch.matmul(M, x).view(output_channels, img.shape[1], img.shape[2])
        
        cS = torch.cumsum(S, dim=-1)
        meanvar = torch.mean(cS[output_channels-1]/cS[ -1])
        # print(f"mean variance with {output_channels} PCs: {meanvar}")

        reduced = reduced.detach().cpu().numpy()
        u = np.mean(reduced, axis=(1, 2), keepdims=True)
        mx = np.amax(reduced, axis=(1, 2), keepdims=True)
        mn = np.amin(reduced, axis=(1, 2), keepdims=True)

        out = (reduced - mn)/(mx - mn)

        return out

    def pca_compute(self, input_images, output_channels=3):
        """ Computes the PCA matrix on a folder of images
        @Params:
            input_images (str): Filepath to folder containing images
            output_channels (int): Number of output channels

        @Returns:
            PCA_mat (torch.tensor[output_channels, input_channels]): PCA Matrix
        """

        PCA_mat = 0
        S_mean = 0
        img_paths = self.all_img_extensions(input_images)
        for i in tqdm(range(len(img_paths)), "Computing PCA Matrix"):
            img = self.read_image(img_paths[i])
            x = torch.from_numpy(img)
            x = x.view(img.shape[0], -1)
            u = torch.mean(x, dim=-1, keepdim=True)
            s = torch.std(x, dim=-1, keepdim=True)
            z = (x-u)/(s + 1e-6)
            cov = torch.matmul(z, torch.transpose(z, 0, 1)) / x.shape[-1]
            
            # print(cov.shape)
            U, S, Vh = torch.svd(cov)
            # print(U.shape, S.shape, Vh.shape)
            
            M = torch.transpose(U[:, :output_channels], 0, 1)

            PCA_mat = i/(i+1) * PCA_mat + 1/(i+1) * M
            S_mean = i/(i+1) * S_mean + 1/(i+1) * S

        cS = torch.cumsum(S_mean, dim=-1)
        meanvar = torch.mean(cS[output_channels-1]/cS[ -1])
        print(f"mean variance with {output_channels} PCs: {meanvar}")

        return PCA_mat

    def pca_reduce(self, img, PCA_mat):
        """ Uses a PCA Matrix to reduce the dimensions of an image

        @Params:
            img (np.ndarray[C, H, W]): input image raster
            PCA_mat (torch.tensor[out_channels, in_channels]):
                                    PCA Matrix computed using 
                                    self.pca_compute
        
        @Returns:
            out (np.ndarray[C', H, W]): output image raster, where
                                    C' == out_channels
        """
        x = torch.from_numpy(img).view(img.shape[0], -1)
        reduced = torch.matmul(PCA_mat, x).view(-1, img.shape[1], img.shape[2])

        reduced = reduced.detach().cpu().numpy()
        u = np.mean(reduced, axis=(1, 2), keepdims=True)
        mx = np.amax(reduced, axis=(1, 2), keepdims=True)
        mn = np.amin(reduced, axis=(1, 2), keepdims=True)

        out = (reduced - mn)/(mx - mn + 1e-6)
        return out


    def preprocess(self, 
            src_imgs, 
            src_labels, 
            dest_imgs, 
            dest_labels, 
            augs = [],#rand_shear, rand_shear, rand_rotate, rand_rotate, rand_flip, rand_flip, rand_colours, rand_brightness, rand_all, rand_all, rand_all, rand_all]
            img_size = (128, 128)):

        """ Preprocesses the images and labels by reshaping the images to 
        img_size, and updating the labels with the bounding boxes reshaped
        to match the new image shape.

        Arguments:
            src_imgs (string): the path to the folder containing original images
            src_labels (string): path to the folder containin original xml labels
            dest_imgs (string): path to folder to save the final images
            dest_labels (string): path to folder to save the final labels.
        """

        all_labels = glob.glob(os.path.join(src_labels, "*.xml"))

        for m in tqdm(range(len(all_labels)), "Preprocessing Images"):


            root = ET.parse(all_labels[m]).getroot()

            img_name = root.find("filename").text

                

            old_width = int(root.find("size").find("width").text)
            old_height = int(root.find("size").find("height").text)
            new_width, new_height = img_size

            bbox_list = []
            for obj in root.findall("object"):
                old_xmin = int(float(obj.find("bndbox").find("xmin").text))
                old_xmax = int(float(obj.find("bndbox").find("xmax").text))
                old_ymin = int(float(obj.find("bndbox").find("ymin").text))
                old_ymax = int(float(obj.find("bndbox").find("ymax").text))
                
                new_xmin = int(old_xmin * new_width/old_width)
                new_ymin = int(old_ymin * new_height/old_height)
                new_xmax = int(old_xmax * new_width/old_width)
                new_ymax = int(old_ymax * new_height/old_height)

                obj.find("bndbox").find("xmin").text = str(new_xmin)
                obj.find("bndbox").find("xmax").text = str(new_xmax)
                obj.find("bndbox").find("ymin").text = str(new_ymin)
                obj.find("bndbox").find("ymax").text = str(new_ymax)

                bbox_list.append(BoundingBox(new_xmin, new_ymin, new_xmax, new_ymax))

            bboxes = BoundingBoxesOnImage(bbox_list, img_size)

            root.find("size").find("width").text = str(img_size[0])
            root.find("size").find("height").text = str(img_size[1])

            src_img_path = os.path.join(src_imgs, img_name)

            # saves unaugmented image
            # old_img = Image.open(src_img_path)
            # new_img = old_img.resize(img_size)
            # new_img.save(os.path.join(dest_imgs, img_name))

            old_img = self.read_image(src_img_path)
            new_img = self.resize_image(old_img, img_size)
            ## TODO: Channel Concatenation with Indices ##

            self.write_image(new_img, os.path.join(dest_imgs, img_name))


            xml_str = ET.tostring(root)
    
            with open(os.path.join(dest_labels, img_name[:-4] + ".xml"), "wb") as f:
                f.write(xml_str)


            for i, x in enumerate(augs):
                aug_img, aug_root = x(new_img, bboxes, deepcopy(root), i)
                aug_xml_str = ET.tostring(aug_root)
                aug_img_name = aug_root.find("filename").text

                with open(os.path.join(dest_labels, aug_img_name[:-4] + ".xml"), "wb") as f:
                    f.write(aug_xml_str)
                # aug_img.save(os.path.join(dest_imgs, aug_img_name))
                self.write_image(aug_img, os.path.join(dest_imgs, aug_img_name))
            
        
    def raster_to_numpy(self, raster):
        """ Converts a tiff raster into a numpy array. Returns 
        Numpy Array that is normalized from 0 to 1 based on 
        the array's datatype
        
        @Params:
            raster (Dataset): the tiff raster
            
        @Returns:
            (np.ndarray[C, H, W]): A numpy array in channel-first format
        """

        np_img = raster.ReadAsArray()
        dtype = np_img.dtype
        # print(dtype)
        max_val = np.iinfo(dtype).max
        return (raster.ReadAsArray() / max_val).astype(np.float32)
    
    
    
    def read_image(self, path):
        """
        Reads an n-channel image into a numpy array.
        Works with PNG, JPEG, and TIFF images.
        
        @Params:
            path (string): The path to the image
            
        @Returns:
            (np.array): The numpy array of the image.
        """
        if path.lower().endswith((".png", "jpeg", "jpg")):
            img = self.read_reg(path)
            img = np.array(img)
            maxval = np.iinfo(img.dtype).max
            #converts to (C, H, W)
            return np.moveaxis(img, -1, 0) / max(maxval, 1)
            
        elif path.lower().endswith((".tif", ".tiff")):
            raster, gt, proj = self.read_tiff(path)
            return self.raster_to_numpy(raster)
    
    def read_tiff(self, path):
        """
        Reads a TIFF raster, then outputs the
        GDAL dataset (the raster along with 
        its transformations
        
        @Params:
            path (string): the path to the .tif image
            
        @Returns:
            raster (gdal.Dataset): the TIFF raster
            geotransform: The geotransformation of the TIFF.
                            Contains the spatial data of 
                            the GeoTIFF.
            proj: The projection scheme used by the TIFF
        """
        raster = gdal.Open(path)
        geotransform = raster.GetGeoTransform()
        proj = raster.GetProjection()
        return raster, geotransform, proj

    def read_reg(self, path):
        """ Read Regular image (png, jpeg)
        
        @Params: 
            path (string): Path to the jpg or png image
            
        @Returns:
            (np.ndarray[H, W, C]): A numpy array in channels-last format
        
        """
        img = Image.open(path)
        return np.array(img)
    
    def read_xml(self, path):
        """ Reads an XML file and returns labels.
        The XML file must be in PASCAL VOC format
        
        @Params:
            path (string): path to the xml file.
            
        @Returns:
            items ([dict]): A list of dictionaries, where each 
                    dictionary represents a bounding box prediction
                    on the image. Each dictionary has these keys:
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "filename",
                        "class",
                        "conf"
        """
        
        root = ET.parse(path).getroot()
        filename = root.find("filename").text
        
        items = []
        # loop through all bounding boxes
        for obj in root.findall("object"):
            attribs = {
                "filename": filename,
                "class": obj.find("name").text, 
                "conf": obj.find("pose").text
            }
            
            bbox = obj.find("bndbox")
            
            attribs["x1"] = int(bbox.find("xmin").text)
            attribs["x2"] = int(bbox.find("xmax").text)
            attribs["y1"] = int(bbox.find("ymin").text)
            attribs["y2"] = int(bbox.find("ymax").text)
            
            items.append(attribs)
            
        return items
    
    
        
    def rebuild_original(self, save_path, all_labels):
        """ Turns all the tiled labels back into a single file.
        Note that it is crucial that the all_labels list contains 
        tile predictions that are in the SAME ORDER as the windows.
        
        In other words, to make a prediction on a large image, first tile the 
        large image into smaller images, then feed the small images into the NN
        WITHOUT changing its order, then obtain the predictions WITHOUT changing its 
        order, and finally rebuild the full prediction using this function.
        
        @Params:
            save_path (string): place to save the rebuilt-labels
            all_labels ([[dict]]): a list containing lists of dictionaries.
                                    The inner list holds the bounding boxes of one 
                                    tile, which are represented by dictionaries.
                                    
        @Returns:
            rebuilt_labels ([dict]): A list of dictionaries, where each 
                                    dictionary represents a bounding box prediction
                                    on the image. Each dictionary has these keys:
                                        "x1",
                                        "y1",
                                        "x2",
                                        "y2",
                                        "filename",
                                        "class",
                                        "conf"
                                    Note that we went from a [[dict]] 
                                    (representing multiple images with their
                                    respective labels) to [dict] (single image, 
                                    containing all the labels.)
        """
        rebuilt_labels = []
        
        for i, window in enumerate(self.windows):
            local_label = all_labels[i]
            x = window.x
            y = window.y
            
            for pred in local_label:
                new_pred = deepcopy(pred)
                new_pred["x1"] = pred["x1"] + x
                new_pred["y1"] = pred["y1"] + y
                new_pred["x2"] = pred["x2"] + x
                new_pred["y2"] = pred["y2"] + y
                
                rebuilt_labels.append(new_pred)
                
        self.write_xml(rebuilt_labels, "test", save_path, 0, 0, 3)
                
        return rebuilt_labels
                
            
    
    def resize_image(self, img, img_size=(128, 128)):
        """ N-channel image resize using bilinear interpolation.
        Credits to this site:
            https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html
        """


        channels, old_height, old_width = img.shape
        new_height, new_width = img_size

        new_img = np.zeros((channels, new_height, new_width))

        for channel in range(img.shape[0]):

            img_channel = img[channel, :, :].ravel()

            x_ratio = float(old_width - 1) / (new_width - 1) if new_width > 1 else 0
            y_ratio = float(old_height - 1) / (new_height - 1) if new_height > 1 else 0

            y, x = np.divmod(np.arange(new_height * new_width), new_width)

            x_l = np.floor(x_ratio * x).astype('int32')
            y_l = np.floor(y_ratio * y).astype('int32')

            x_h = np.floor(x_ratio * x).astype('int32')
            y_h = np.floor(y_ratio * y).astype('int32')

            x_weight = (x_ratio * x) - x_l
            y_weight = (y_ratio * y) - y_l

            a = img_channel[y_l * old_width + x_l]
            b = img_channel[y_l * old_width + x_h]
            c = img_channel[y_h * old_width + x_l]
            d = img_channel[y_h * old_width + x_h]

            resized = a * (1 - x_weight) * (1 - y_weight) + \
                        b * x_weight * (1 - y_weight) + \
                        c * y_weight * (1 - x_weight) + \
                        d * x_weight * y_weight

            # print(resized.shape, new_height*new_width)
            new_img[channel] = resized.reshape(new_height, new_width)

        return new_img





    def tile_image(self, img, tile_size, labels=None, savepath=None, overlap=0.1):
        """
        Splits an image along with its bounding boxes into 
        tiles. If labels are not specified, then tiling will
        only apply to the images. Labels will be stored under 
        each sliding window.
        
        @Params:
            img (np.ndarray[C, H, W]): The input image array in
                                channel-first format.
            tile_size (int): Size of the sliding window 
            labels ([dict]): A list of dictionaries, where each 
                                dictionary represents a bounding box prediction
                                on the image. Each dictionary has these keys:
                                    "x1",
                                    "y1",
                                    "x2",
                                    "y2",
                                    "filename",
                                    "class",
                                    "conf"
                                Labels can be obtained through self.read_xml()
            savepath (string): savepath to folder for image tiles only. Tile annotations
                            are stored in each self.windows[i]
            overlap (float): The percentage of overlap between sliding windows
        
        @Returns:
            im_arrays ([np.ndarray[C, H, W]]): A list of numpy arrays representing 
                                the tiled images.
            self.windows ([slidingwindow]): A list of windows 
        """
        
        self.windows = self.get_tiles(img, tile_size, overlap)
        im_arrays = []
        
        # For each window, find bounding boxes that are local to it, 
        # then transform the bounding box coords so that it is local 
        # to this window
        for i in tqdm(range(len(self.windows)), desc="Tiling image:"):
            xmin = self.windows[i].x
            ymin = self.windows[i].y
            xmax = xmin + self.windows[i].w
            ymax = ymin + self.windows[i].h
            
            if labels is not None:
                self.windows[i].labels = self.find_local_labels(self.windows[i].x, self.windows[i].y, self.windows[i].w, self.windows[i].h, labels)

            #Crop tile
            if savepath is None:
                im_arrays.append(img[:, ymin:ymax, xmin:xmax])
            else:
                c = img.shape[0]
                if c > 3:
                    self.write_image(img[:, ymin:ymax, xmin:xmax], path=os.path.join(savepath, str(i) + ".tif"))
                else:
                    self.write_image(img[:, ymin:ymax, xmin:xmax], path=os.path.join(savepath, str(i) + ".png"))


            
        return (im_arrays, self.windows)
            
    
    def write_image(self, img, path, geotransform=None, projection=None):
        """ Writes an np.array image into a file.
        Supports writing JPEGs, PNGs, and TIFFs.
        
        @Params:
            img (np.ndarray[C, H, W]): A np.array in the channel-
                                    first format
            path (string): Save path to the image
        """
        
        if path.lower().endswith((".png", "jpeg", "jpg")):
            img = np.moveaxis(img, 0, -1)
            self.write_reg(img, path)
            
        elif path.lower().endswith((".tif", ".tiff")):
            self.write_tiff(img, path, geotransform, projection)
    
    def write_reg(self, img_array, path):
        """ Writes PNG and JPG images
        @Params:
            img_array (np.ndarray[H, W, C]): A numpy array in the channel-
                                    last format
            path (string): Save path of the image. Must have file extensions
                                    of PNG or JPG (or any of its variations)
        """
        img_array = (img_array * 255).astype(np.uint8)
        im = Image.fromarray(img_array)
        im.save(path)
        
    def write_tiff(self, img_array, path, geotransform=None, projection=None):
        """ Writes TIFF rasters.
        Note: if geotransform or projection are not specified, then the output 
        will not be displayable by image-viewing software. The raster, however, 
        still contains the data.

        @Params:
            img_array (np.ndarray[C, H, W]): A numpy array in the channel-
                                    last format
            path (string): Save path of the image. Must have file extensions
                                    of TIFF (or any of its variations)
        """
        self.numpy_array_to_raster(path, img_array, geotransform, projection)
        
        
    def write_xml(self, labels, img_path, label_path, w, h, c=3):
        """ Writes labels into XML files in the PASCAL VOC format
        
        @Params:
            labels ([dict]): A list of dictionaries, where each 
                    dictionary represents a bounding box prediction
                    on the image. Each dictionary has these keys:
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "filename",
                        "class",
                        "conf"
            img_path (string): path to the folder containing tiled images
            label_path (string): save path of the XML file
            w (int): width of image
            h (int): height of image
        """
        writer = Writer(img_path, w, h, c)
        
        for label in labels:
            
            name = label["class"]
            xmin = label["x1"]
            ymin = label["y1"]
            xmax = label["x2"]
            ymax = label["y2"]
            conf = label["conf"]
            
            writer.addObject(name, xmin, ymin, xmax, ymax, conf)
            
        writer.save(label_path)
              






























class Color():
    """
    This class defines the colours for labeling
    the class of predictions.
    """
    def __init__(self):
        self.colours = [ "red", 
                    "orange", 
                    "yellow", 
                    "green", 
                    "blue", 
                    "violet",
                    "gray", 
                    "white" ]
        random.shuffle(self.colours)
    def __getitem__(self, idx):
        return self.colours[idx % len(self.colours)]
        


