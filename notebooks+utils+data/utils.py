import h5py
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def get_img_boxes(f, idx=0):
    """
    Get the 'height', 'left', 'top', 'width', 'label' of bounding boxes of an image
    Adapted from https://www.vitaarca.net/post/tech/access_svhn_data_in_python/

    Args
        f: digitStruct.mat h5py file
        idx: index of the image
        
    Returns:
        dictionary of bounding box values as integers
    """
    bboxs = f['digitStruct/bbox']
    box = f[bboxs[idx][0]]
    meta = { key : [] for key in box.keys()}

    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))

    return meta
    
def get_img_name(f, idx=0):
    """
    Get the name of an image given it's index.
    Adapted from https://www.vitaarca.net/post/tech/access_svhn_data_in_python/

    Args
        f: digitStruct.mat h5py file
        idx: index of the image
        
    Returns:
        image name as a string
    """
    names = f['digitStruct/name']
    img_name = ''.join(map(chr, f[names[idx][0]][()].flatten()))
    return(img_name)



def create_annot_file(f, path, idx=0):
    """
    Create a single Darknet TXT annotation file for an image.
    Writes to file <image name>.txt in same directory as image.

    Args
        f: digitStruct.mat h5py file
        path: path: path to digitStruct.mat
        idx: index of the image
        
    Returns:
        None
    """
    # get image name and bounding info
    name = get_img_name(f, idx)
    boxes = get_img_boxes(f, idx)
    
    # get dimensions of image
    try:
        (h_img, w_img) = cv.imread(path + name).shape[:2]
    except:
        print(f"ERROR: Could not open {name} to get dimensions.")
        print("Make sure image is in same directory as digitStruct.mat")
        print(f"Tried:  {path + name}")
        
    # initialize list for annotations
    annots = []
    
    for i in range(len(boxes['label'])):
        # get original bounding values
        (x, y) = (boxes['left'][i], boxes['top'][i])
        (w, h) = (boxes['width'][i], boxes['height'][i])

        # transform x and y
        centerX = x + (w / 2)
        centerY = y + (h / 2)

        # normalize bounding values
        centerX /= w_img
        centerY /= h_img
        w /= w_img
        h /= h_img

        # get label
        label = boxes['label'][i] if boxes['label'][i] != 10 else 0

        # append annotation in Darknet format to annotation list
        annots.append(f'{label} {centerX} {centerY} {w} {h}\n' )
    
    # write annotations to file 
    annot_file = open(path + name.split('.')[0] + '.txt', 'w')
    annot_file.writelines(annots)
    annot_file.close()

def create_annot_files(path):
    """
    Create Darknet TXT annotation file for all images in directory.
    Writes to files <image name>.txt in same directory as images.

    Args
        path: path to digitStruct.mat
        
    Returns:
        None
    """
    if path[-1] != '/':
        path += '/'
    
  #  try:
        f = h5py.File(f'{path}digitStruct.mat', mode='r')
  #  except:
   #     print("ERROR: Could not open file.  Check path to digitStruct.mat")
        
    for i in range(len(f['digitStruct/name'])):
        create_annot_file(f, path, i)

#define utility function
def imShow(path):
    """
    Show image in directory.

    Args
        path: path to digitStruct.mat
        
    Returns:
        Image
    """

    image = cv.imread(path)
    height, width = image.shape[:2]
    resized_image = cv.resize(image,(3*width, 3*height), interpolation = cv.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv.cvtColor(resized_image, cv.COLOR_BGR2RGB))
    plt.show()

def get_cropped_bib(image, input_path, out_path):  
    """
    Read in the RBNR image bounding box information and use it to save
    cropped out images of the bibs in the original image.  Then write
    the cropped bib image name and RBN to file.
    
    Args
        image (str): name of original image
        input_path (str): path to directory of image
        out_path (str): directory where results are saved
        
    Returns
        None
    """
    
    #load image
    img = cv.imread(input_path + image)
    
    # load annotation file
    f = sio.loadmat(input_path + image + '.mat')

    #get bounding boxes and bib numbers
    boxes = f['tagp']
    numbers = f['number'].flatten()

    for i, box in enumerate(boxes):
        #convert box values to int
        (t, b, l, r) = [int(i) for i in box]
        # crop image and save
        crop_img = img[t:b, l:r]
        crop_name = image[:-4]+'_'+'bib_'+str(i+1)+'.JPG'
        cv.imwrite(out_path + crop_name, crop_img)
        # write race bib number to file
        rbn_file = open(out_path + 'bib_numbers.txt', 'a')
        rbn_file.writelines(f"{crop_name},{numbers[i]}\n")

def create_labeled_image(image, input_path, out_path, configPath, weightsPath): 
    """
    Run digit detection and save a labeled image.  Then compile digits
    into single RBN and save to file for validation.
    Code for using YOLO in OpenCV adapted from OpenCV Docs:
    https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
    
    Args
        image (str): name of original image
        input_path (str): path to directory of image
        out_path (str): directory where results are saved
        
    Returns
        None
    """
    # get random colors for boxes
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(10, 3), dtype='int64')

    # Give the configuration and weight files for the model to load into the network.
    net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

    # determine the output layer(s)
    ln = net.getLayerNames()
    ln = [ln[i- 1] for i in net.getUnconnectedOutLayers()]
    # read in image and construct a blob from the image
    img = cv.imread(input_path + image)
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

    # get detections
    net.setInput(blob)
    outputs = net.forward(ln)

    # initialize lists
    boxes = []
    confidences = []
    classIDs = []
    
    # initialize image dimensions
    h_img, w_img = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Only keep detection if it is for a digit with high confidence
            if confidence > 0.5:
                box = detection[:4] * np.array([w_img, h_img, w_img, h_img])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    # get indices of final bounding boxes  
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # initialize list for digit position and value
    bib_digit_loc = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            
            cv.rectangle(img, (x, y), (x + w, y + h), color, 1)
            text = "{}: {:.4f}".format(classIDs[i], confidences[i])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            bib_digit_loc.append((x, str(classIDs[i])))
        
        # save annotated image
        cv.imwrite(out_path+image[:-4]+'_'+'detected'+'.JPG', img)
        
        # write race bib number to file
        bib_digit_loc.sort()
        rbn_pred = int(''.join([i[1] for i in bib_digit_loc]))
        #orig_image = '_'.join(image.split('_')[:2]) + '.JPG'
        rbn_pred_file = open(out_path + 'rbn_preds.txt', 'a')
        rbn_pred_file.writelines(f"{image},{rbn_pred}\n")

def get_bbox_obj(path, img_name, shape):
    """
    Get RBNR bounding box info for a given image.
    
    Args
        path (str): path to image directory
        img_name (str): image name
        shape: (width, height) of image
        
    Returns
        Bounding Boxes object containing all boxes for image in the format
        (x1, y1, x2, y2)
    """
    # load annotation file
    f = sio.loadmat(path + img_name + '.mat')
    
    # get bounding values
    boxes = f['tagp']
    # create bounding boxes object
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=box[2], y1=box[0], x2=box[3], y2=box[1]) for box in boxes
    ], shape=shape)
    
    return bbs

def augment(images, bbs):
    """
    Apply random agumentations to a list of images and bounding boxes.
    
    Args
        images (list of numpy arrays): list of images from openCV .imread
        bbs (list of BoundingBoxesOnImage): list bounding box objects
        
    Returns
        augmented images and bounding boxes in separate lists
    """
    seq = iaa.Sequential([
        #iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order
    
    return seq(images=images, bounding_boxes=bbs)

def create_annot_file_bbs(bbs, img, img_name, path):
    """
    Create a single Darknet TXT annotation file for an image.
    Writes to file <image name>.txt in same directory as image.

    Args
        bbs (BoudningBoxesOnImage): bounding boxes object for the image
        img (numpy array): image as numpy array from openCV .imread
        img_name (str): name of image
        path (str): path to image directory
        
    Returns
        None
    """
    
    # get dimensions of image
    (h_img, w_img) = img.shape[:2]
        
    # initialize list for annotations
    annots = []
    
    for i in range(len(bbs.bounding_boxes)):
        box = bbs.bounding_boxes[i]
        # get original bounding values
        (x, y) = (box.x1, box.y1)
        (w, h) = (box.x2 - box.x1, box.y2 - box.y1 )

        # transform x and y
        centerX = x + (w / 2)
        centerY = y + (h / 2)

        # normalize bounding values
        centerX /= w_img
        centerY /= h_img
        w /= w_img
        h /= h_img

        # append annotation in Darknet format to annotation list
        annots.append(f'{0} {centerX} {centerY} {w} {h}\n' )
    
    # write annotations to file 
    annot_file = open(path + img_name.split('.')[0] + '.txt', 'w')
    annot_file.writelines(annots)
    annot_file.close()

def create_augmented_images(num, input_path, img_name, output_path):
    """
    Create a given number of augmented images with bounding boxes given an
    original image.
    
    Args
        num (int): number of augmented images to create
        input_path (str): path to original image directory
        img_name (str): name of original image
        output_path (str): directory for saving augmented images and bounding
                           box files
                            
    Returns
        None
    """
    
    # Duplicate image and resize
    # This size is the size of the YOLO model input
    images = [cv.imread(input_path + img_name) for _ in range(num)]

    # bring in bounding boxes
    bbs_list = [get_bbox_obj(input_path, img_name, images[0].shape) for _ in range(num)]
    
    # augment
    images_aug, bbs_aug = augment(images, bbs_list)
    
    # save augmented images and bounding boxes
    for i, img in enumerate(images_aug):
        img_filename = img_name[:-4] + '_' + str(i+1) + '.JPG'
        cv.imwrite(output_path + img_filename, img)
        boxes = bbs_aug[i].remove_out_of_image().clip_out_of_image()
        create_annot_file_bbs(boxes, img, img_filename, output_path)

def create_test_images(num, input_path, img_name, output_path):
    """
    Create a given number of augmented images with bounding boxes given an
    original image.
    
    Args
        num (int): number of augmented images to create
        input_path (str): path to original image directory
        img_name (str): name of original image
        output_path (str): directory for saving augmented images and bounding
                           box files
                            
    Returns
        None
    """
    
    # Duplicate image and resize
    # This size is the size of the YOLO model input
    images = [cv.imread(input_path + img_name) for _ in range(num)]

    # bring in bounding boxes
    bbs_list = [get_bbox_obj(input_path, img_name, images[0].shape) for _ in range(num)]
    
    # augment
   # images_aug, bbs_aug = augment(images, bbs_list)
    
    # save augmented images and bounding boxes
    for i, img in enumerate(images):
        img_filename = img_name[:-4] + '_' + str(i+1) + '.JPG'
        cv.imwrite(output_path + img_filename, img)
        boxes = bbs_list[i].remove_out_of_image().clip_out_of_image()
        create_annot_file_bbs(boxes, img, img_filename, output_path)

def darknet_to_standard(path, f_name, img):
    """
    Read in bounding box values for an image from Darknet formated annotation
    file, and convert for drawing on image.
    
    Args
        path (str): path to directory of image
        f_name (str): name of image file
        img (numpy array): image array from openCV .imread
        
    Returns:
        list of bounding boxes for image as [x, y, width, height]
    """
    
    # get original image dimension
    (h_img, w_img) = img.shape[:2]
    
    #read in bounding box from Darknet file
    f = open(path + f_name)
    objects = f.readlines()
    
    boxes = []
    for obj in objects:
        # get bounding box values
        box = [float(i) for i in obj.strip().split()[1:]]
        # convert from normalized to original size
        sized_box = box * np.array([w_img, h_img, w_img, h_img])
        # convert x and y from center to corner
        (centerX, centerY, width, height) = sized_box.astype("int")
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        # reconfigure and save to list
        orig_box = [x, y, int(width), int(height)]
        boxes.append(orig_box)
    
    return boxes
        
def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))  

class Detector:
    """
    Create YOLO object detection model in OpenCV with a given config and weights.
    Use this model to make predictions.
    
    Attributes
        classes (list): list of class names
        net (obj): openCV network object
        ln (obj): openCV layer names object
    """
    
    def __init__(self, cfg, wts, classes):
        """Initialize detector object
        
        Args
            cfg (str): path to model config file
            wts (str): path to model weights file
            classes (list): list of class names
        """
        
        self.classes = classes
        self.net = cv.dnn.readNetFromDarknet(cfg, wts)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

        # determine the output layer
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i-1] for i in self.net.getUnconnectedOutLayers()]
        
    def detect(self, img, conf):
        """
        Make predictions and return classes and bounding boxes
        
        Args
            img (numpy array): image array from openCV .imread
            conf (float): prediction confidence threshold
            
        Returns
            List containing bounding box values and class names for detections
            in the form [<class name>, [x, y, width, height]]
        """
        
        #format image for detection
        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        
         # get detections
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)

        # initialize lists
        boxes = []
        confidences = []
        classIDs = []

        # initialize image dimensions
        h_img, w_img = img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # drop low confidence detections and 
                if confidence > conf:
                    box = detection[:4] * np.array([w_img, h_img, w_img, h_img])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non maximal suppression for
        # initialize lists
        self.boxes = []
        self.confidences = []
        self.detected_classes = []
        cls_and_box = []
        # get indices of final bounding boxes  
        indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                self.boxes.append(boxes[i])
                self.confidences.append(confidences[i])
                self.detected_classes.append(self.classes[classIDs[i]])
                
                cls_and_box.append([self.classes[classIDs[i]], boxes[i]])
        
        return cls_and_box

def get_rbns(img, bd_configPath, bd_weightsPath,bd_classes,nr_configPath,nr_weightsPath,nr_classes,single=False):
    """
    Given an image return bib numbers and bib bounding boxes for detected bibs
    
    Args
        img (numpy array): image array given by openCV .imread
        single (bool): whether one or many bib detections will be
            returned.  If true, return detection with largest bounding
            box area.
            
    Returns
        List of detected bib numbers and corresponding bounding boxes in
        the format [<bib number>, [x, y, width, height]]
    """
    
    # Instantiate detectors
    bd = Detector(bd_configPath, bd_weightsPath, bd_classes)
    nr = Detector(nr_configPath, nr_weightsPath, nr_classes)

    # Make bib location predictions
    bib_detections = bd.detect(img, 0.25)


    if len(bib_detections) > 0:
        for obj in bib_detections:
            # crop out detected bib
            (x, y, w, h) = obj[1]
            obj.append(w * h)
            crop_img = img[y:y+h, x:x+w]
            
            # detect numbers on bib
            num_detections = nr.detect(crop_img, 0.5)
            bib_digit_loc = []
            if len(num_detections) > 0:
                # get digits and locations
                for digit in num_detections:
                    (d_x, d_y, d_w, d_h) = digit[1]
                    bib_digit_loc.append((d_x, str(digit[0])))

                # sort detected numbers L->R and put together
                bib_digit_loc.sort()
                rbn = int(''.join([i[1] for i in bib_digit_loc]))
                obj.append(rbn)
            else:
                obj.append(0) # bib detection but no digit detection

        if single: 
            if len(bib_detections) > 1:
                bib_detections.sort(key=lambda x: x[2], reverse=True)
            return [[bib_detections[0][3], bib_detections[0][1]]]
        else:
            final_bibs = []
            for bib in bib_detections:
                final_bibs.append([bib[3], bib[1]])
            return final_bibs
    else: return None

def get_true_annot(image, input_path, out_path):  
    """
    Read in the RBNR annotation file and return annotations
    
    Args
        image (str): name of original image
        input_path (str): path to directory of image
        out_path (str): directory where results are saved
        
    Returns
        List of annotations in format 
        [[<bib number>, [x, y, width, height]]]
    """
    
    # load annotation file
    f = sio.loadmat(input_path + image + '.mat')

    #get bounding boxes and bib numbers
    boxes = f['tagp']
    numbers = f['number'].flatten()

    bib_annots = []
    for i, box in enumerate(boxes):
        #convert box values to int
        (y1, y2, x1, x2) = [int(i) for i in box]
        
        # add rbn and formated bounding box to list
        bib_annots.append([numbers[i], [x1, y1, x2-x1, y2-y1]])
        
        # add true bib numbers to file
        true_file = open(out_path + 'bib_numbers.txt', 'a')
        true_file.writelines(f"{image},{numbers[i]}\n")
        true_file.close()
    
    return bib_annots

def show_local_mp4_video(file_name, width=640, height=480):
  import io
  import base64
  from IPython.display import HTML
  video_encoded = base64.b64encode(io.open(file_name, 'rb').read())
  return HTML(data='''<video width="{0}" height="{1}" alt="test" controls>
                        <source src="data:video/mp4;base64,{2}" type="video/mp4" />
                      </video>'''.format(width, height, video_encoded.decode('ascii')))