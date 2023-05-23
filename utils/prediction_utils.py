
"""
prediction_utils.py

This module contains functions used to process 
YOLOv5 raw outputs, most notably the predict()
function.

Author: Zony Yu
"""



import os
import sys

import torch
import torchvision
import numpy as np
from tqdm import tqdm
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET

from utils.general import non_max_suppression

from utils.image_processor import Image_processor
from preprocessing import idx_to_name, name_to_idx







def predict(model, input_img_path, PCA_mat=None, thresh=0.3, nms_thresh=0.2, scale_down=1):
    '''
    YOLOv5 prediction pipeline for this workflow.
    Designed to work with very large images by tiling
    the image down to 1600 x 1600 px tiles and predicting
    on batch.

    @Params:
        model: YOLOv5 model. This is the output of DetectMultiBackend()
                from models.common. An example implementation is in
                predict.py.
        input_img_path (str): Path to a single image
        PCA_mat (np.ndarray[K, N]): PCA Matrix, where K is the output dimensions
                                (In case of images, it would be the channels), 
                                and N is the input dimensions.
        thresh (float): Confidence threshold for filtering. All confs > 0.3 will 
                                be kept.
        nms_thresh (float): Non-Max-Suppression threshold
        scale_down (float): The factor of which the image is to be scaled down.
                            Only use this if the model was trained on scaled down
                            images.

    @Returns:
        boxes (torch.tensor[N, 4]): The set of bounding boxes in the image, where
                                    N is the number of bounding boxes, and 
                                    each bounding box constists of [x1, y1, x2, y2]
                                    representing the top left and bottom right corners 
                                    of the box.
                                    
                                        [..., [x1, y1, x2, y2], ...]

        confs (torch.tensor[N]): The confidence values of each bounding box.

        clss (torch.tensor[N]): the label-encoded (i.e. not one-hot encoded) 
                                    class prediction for each bounding box.
    '''

    # Set GPU 
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    device = torch.device(dev)  

    # Read image
    ip = Image_processor()
    im = ip.read_image(input_img_path)
    print(f'\n\nshape = {im.shape}\n\n')
    c, h, w = im.shape

    nn_img_size = 1600

    # tile images and resize if necessary
    im_array, _ = ip.tile_image(im, tile_size=scale_down*nn_img_size, overlap=0.3)
    small_im = [ip.resize_image(x, img_size=(nn_img_size, nn_img_size)) for x in im_array] if scale_down > 1 else im_array
    im_array = np.array(small_im)


    batch_size = 4
    pred_idx = 0

    use_half = False
    for param in model.parameters():
        if param.dtype == torch.float16:
            use_half = True
            break


    all_boxes = []
    # Loop through batches
    for i in tqdm(range(int(np.ceil(len(im_array)/batch_size))), "Predicting on Batch"):

        batch = im_array[batch_size*i:batch_size*(i+1)]

        # Use PCA to reduce each image in a batch
        if PCA_mat is not None:
            for j in range(len(batch)):
                batch[j] = ip.pca_reduce(batch[j], PCA_mat=PCA_mat)
        
        batch = np.array(batch)
        if use_half:
            X = torch.from_numpy(batch).to(device=device).half()
        else:
            X = torch.from_numpy(batch).to(device=device).float()
        Y = model(X)

        # Obtain set of boxes for each prediction
        for k in range(batch.shape[0]):

            
            pred = non_max_suppression(torch.unsqueeze(Y[k], dim=0), thresh, nms_thresh, classes=None, agnostic=False, max_det=5000)
           
                


            boxes = torch.zeros(1)
            confs = torch.zeros(1)
            clss = torch.zeros(1)
            # Note: m is always 0, 
            # meaning that there is one det in pred.
            for m, det in enumerate(pred):
                boxes = torch.zeros((len(det), 4))
                confs = torch.zeros((len(det)))
                clss = torch.zeros((len(det)))
                remove = torch.zeros((len(det))).bool()
                j=0
                # each det contains multiple xyxy, conf, cls
                for *xyxy, conf, cls in det:
                    xyxy = torch.tensor([x.item() for x in xyxy])
                    # print(xyxy)
                    boxes[j, :] = xyxy
                    confs[j] = conf
                    clss[j] = cls

                    # remove boxes
                    asp = (torch.abs(xyxy[3] - xyxy[1]))/(torch.abs(xyxy[2]-xyxy[0]))
                    edge = xyxy[0] < 3 or xyxy[1] < 3 or xyxy[2] > nn_img_size - 3 or xyxy[3] > nn_img_size - 3 
                    remove[j] = (asp < 3.0 and asp > 0.33) and not edge
                        

                    j+=1
            boxes[:, :] *= scale_down

            # Remove boxes with long aspect ratios
            boxes = boxes[remove]
            confs = confs[remove]
            clss = clss[remove]

            boxes = convert_boxes(boxes, confs, clss, idx_to_name)
            all_boxes.append(boxes)

    os.system("mkdir .temp")
    ip.rebuild_original(".temp/predictions.xml", all_labels=all_boxes)

    full_boxes, confs, clss = xml_to_boxes(".temp/predictions.xml", name_to_idx=name_to_idx)
    print("TEST")
    # Dummy scores just to make sure boxes removed are
    # only due to overlap, and not due to score.
    scores = torch.ones(full_boxes.shape[0])

    print("Running Final NMS...")
    # print(full_boxes)

    
    keep_idx = torchvision.ops.nms(full_boxes.float(), scores.float(), iou_threshold=0.2)

    #redefinition of boxes
    boxes = full_boxes[keep_idx]
    scores = scores[keep_idx]
    confs = confs[keep_idx]
    clss = clss[keep_idx]

    print("Saving Prediction...")
    boxes_to_xml(boxes, 
                 confs, 
                 clss, 
                 "predictions/predictions_NMS.xml", 
                 input_img_path, 
                 h, w, 
                 idx_to_name=idx_to_name)

    print("Done.")
    return boxes, confs, clss








def boxes_to_xml(boxes, confs, clss, filename, img_name="", h=0, w=0, idx_to_name=None):
    """ 
    Converts YOLOv5 predictions to PASCAL VOC XMLs.
    The XMLs are in PASCAL VOC annotations format
    (as opposed to detection format), and we use the 
    <pose> tag to store the confidence.

    @Params
        boxes (torch.tensor[N, 4]): Bounding boxes in (x1, y1, x2, y2) format
        confs (torch.tensor[N]): Confidences of predictions
        clss (torch.tensor[N]): Predicted class indices
        filename (str): Save name for the XML file
        img_name (str): Name of the image that produced this prediction
        h (int): Height of the image
        w (int): Width of the image
        idx_to_name (list): Maps index to class name.
    """

    #for each image, we write the XML
    writer = Writer(img_name, w, h)
    for m in range(boxes.shape[0]):
        if idx_to_name is not None:
            name = idx_to_name[int(clss[m].item())]
        else: 
            name = "tree"
        x1 = int(boxes[m, 0].item()) 
        y1 = int(boxes[m, 1].item())
        x2 = int(boxes[m, 2].item())
        y2 = int(boxes[m, 3].item())
        conf = confs[m].item()
        writer.addObject(name, x1, y1, x2, y2, pose=conf, )

    writer.save(filename)
        







def convert_boxes(boxes, confs, clss, idx_to_name=None):
    """
    Converts boxes, confs, and classes in torch.tensor format to 
    list of dicts format

    @Params:
        boxes (torch.tensor[N, 4]): Bounding boxes in (x1, y1, x2, y2) format
        confs (torch.tensor[N]): Confidences of predictions
        clss (torch.tensor[N]): predicted class indices
        idx_to_name (dict or list): Maps index (int) to class name (str)

    @Returns:
        list_boxes (list(dict)): List of dictionaries, where each one represent
                                one object. Each object has the following attributes:
                                    x1: int
                                    y1: int
                                    x2: int
                                    y2: int
                                    filename: string
                                    class: string
                                    conf: float

    """

    list_boxes = []
    for i in range(boxes.shape[0]):
        dct = {
            "x1": int(boxes[i, 0].item()),
            "y1": int(boxes[i, 1].item()),
            "x2": int(boxes[i, 2].item()),
            "y2": int(boxes[i, 3].item()),
            "filename" : "",
            "class": idx_to_name[int(clss[i].item())] if idx_to_name is not None else "tree",
            "conf": confs[i].item()
        }
        list_boxes.append(dct)

    return list_boxes








def xml_to_boxes(xml_path, name_to_idx=None):
    """ 
    Converts PASCAL VOC labels to YOLOv5 predictions.
    The XMLs are in PASCAL VOC annotations format
    (as opposed to detection format), and we use the 
    <pose> tag to store the confidence.

    @Params:
        xml_path (str): Path to the XML file
        name_to_idx (dict): Maps class name (str) to index (int)

    @Returns
        boxes (torch.tensor[N, 4]): Bounding boxes in (x1, y1, x2, y2) format
        confs (torch.tensor[N]): Confidences of predictions
        clss (torch.tensor[N]): Predicted class indices
    """


    root = ET.parse(xml_path).getroot()
    boxes = []
    confs = []
    clss = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")

        x1 = int(bndbox.find("xmin").text)
        y1 = int(bndbox.find("ymin").text)
        x2 = int(bndbox.find("xmax").text)
        y2 = int(bndbox.find("ymax").text)
        boxes.append([x1, y1, x2, y2])
        try:
            conf = float(obj.find("pose").text)
        except:
            conf = 1
        confs.append(conf)

        if name_to_idx is not None:
            try:
                cls = obj.find("name").text
                cls = name_to_idx[cls]
                clss.append(cls)
            except: 
                clss.append(-1)
        else: 
            clss = None
            


    boxes = np.array(boxes)
    return torch.from_numpy(boxes), torch.tensor(confs), torch.tensor(clss)

        











