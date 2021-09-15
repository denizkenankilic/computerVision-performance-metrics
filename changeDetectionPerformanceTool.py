# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:08:08 2021

@author: deniz.kilic
"""

import cv2
import numpy as np
from skimage import io
import os
import json
from skimage import measure

def pixel_prepare_data(groundTruthImage, outputMaskImage):
    """
    A Function to check whether sizes of images are equal, if not
    resize the output image with respect to the ground truth image.
    It is assumed that performance is calculated by b&w images. Therefore after
    resizing images are converted to b&w masks.
    """

    image1 = io.imread(groundTruthImage)
    image2 = io.imread(outputMaskImage)

    if image1.shape[0]!=image2.shape[0] or image1.shape[1]!=image2.shape[1]:
        image2 = cv2.resize(image2, (image1.shape[0], image1.shape[1]))
    if image1.ndim == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        image1 = image1
    if image2.ndim == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        image2 = image2

    gTobw1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    image1Converted = gTobw1[1]

    gTobw2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)
    image2Converted = gTobw2[1]

    return image1Converted, image2Converted

def get_object_label(groundTruthImage, regions):
    """A function that takes coordinate array and groundtruth image to
    return corresponding label for object coordinates in relevant image."""

    objectLabel = None
    x, y = regions.coords[0]
    objectLabel = groundTruthImage[x][y]

    return objectLabel


def get_object_properties(groundTruthImage):
     """
     A function that takes several measure properties of labeled image regions.
     """

     labeledRegions = measure.label(groundTruthImage)
     region_props = measure.regionprops(labeledRegions)

     labelList = []
     bboxList = []
     areaList = []
     bboxAreaList = []
     convexImageList = []
     coordinateList = []
     imageList = []


     for i in range(len(region_props)):
          label = get_object_label(groundTruthImage, region_props[i])
          labelList.append(label)
          bboxList.append(region_props[i].bbox)
          areaList.append(region_props[i].area)
          bboxAreaList.append(region_props[i].bbox_area)
          convexImageList.append(region_props[i].convex_image)
          coordinateList.append(region_props[i].coords)
          imageList.append(region_props[i].image)

     return region_props, labelList, bboxList, areaList, bboxAreaList, convexImageList, coordinateList, imageList

def pixel_perf_measure(gt, cm, areaList, coordinateList):
    """
    Pixel-based performance function for a given ground truth and change map data.
    It calculates the number of true positive, false positive, true negative and false negative.
    """

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # fn's of the small objects are discarded
    for k in range(len(areaList)):
         if (gt.shape[0]*gt.shape[1])/1000 < areaList[k]:
                for m in range(coordinateList[k].shape[0]):
                     if cm[coordinateList[k][m][0], coordinateList[k][m][1]]==0 and gt[coordinateList[k][m][0], coordinateList[k][m][1]]!=cm[coordinateList[k][m][0], coordinateList[k][m][1]]:
                        fn += 1
    for i in range(cm.shape[0]):
         for j in range(cm.shape[1]):
              if gt[i, j]==255 and cm[i, j]==255:
                 tp += 1
              if cm[i, j]==255 and gt[i, j]!=cm[i, j]:
                 fp += 1
              if gt[i, j]==0 and cm[i, j]==0:
                 tn += 1

    return tp, fp, tn, fn

def pixel_perf_metrics(tp, fp, tn, fn, performanceMetricType):
     """
     A function that calculates the performance metric for a pixel-based problem.
     """

     n = tp + fp + tn + fn
     pre = ((tp+fp)*(tp+fn) + (fn+tn)*(tn+fp)) / (n*n)
     pcc = (tp + tn) / n
     prec = tp / (tp + fp)
     recall = tp / (tp + fn)
     if performanceMetricType == 'pcc':
          pcc = pcc
          print('Percentage Correct Classification = ', pcc)
          return pcc
     elif performanceMetricType == 'oe':
          oe = fp + fn
          print('Overall Error = ', oe)
          return oe
     elif performanceMetricType == 'kc':
          kc = (pcc - pre) / (1 - pre)
          print('Kappa Coefficient = ', kc)
          return kc
     elif performanceMetricType == 'jc':
          jc = tp / (tp+fp+fn)
          print('Jaccard Coefficient = ', jc)
          return jc
     elif performanceMetricType == 'yc':
          yc = tp / (tp+fp) + tn / (tn+fn) - 1
          print('Yule Coefficient = ', yc)
          return yc
     elif performanceMetricType == 'prec':
          prec = prec
          print('Precision = ', prec)
          return prec
     elif performanceMetricType == 'recall':
          recall = recall
          print('Recall = ', recall)
          return recall
     elif performanceMetricType == 'fmeas':
          fmeas = 2 * prec * recall / (prec+recall)
          print('F-Measure = ', fmeas)
          return fmeas
     elif performanceMetricType == 'sp':
          sp = tn/(tn+fp)
          print('Specificity = ', sp)
          return sp
     elif performanceMetricType == 'fpr':
          fpr = fp/(fp+tn)
          print('False Positive Rate = ', fpr)
          return fpr
     elif performanceMetricType == 'fnr':
          fnr = fn/(tn+fp)
          print('False Negative Rate = ', fnr)
          return fnr
     elif performanceMetricType == 'pwc':
          pwc = 100*(fn+fp)/(tp+fn+fp+tn)
          print('Percentage of Wrong Classifications = ', pwc)
          return pwc
     else:
          print("Warning: Pixel-based supported metrics are 'pcc', 'oe', 'kc', 'jc', 'yc', 'prec', 'recall', 'fmeas', 'sp', 'fpr', 'fnr', 'pwc'.")
          return None

def append_to_json(jsonStruct, classNum, idxNum, modified, xMin, yMin, xMax, yMax):
    jsonStruct['samples'].append({
        'class': classNum,
        'idx': idxNum,
        'isModified': modified,
        'xMin': xMin,
        'yMin': yMin,
        'xMax': xMax,
        'yMax': yMax
    })
    return jsonStruct

def bbox_prepare_data(groundTruthImage, outputMaskImage):
    """
    A function to check whether sizes of images are equal, if not
    resize the output json file with respect to the ground truth json file.
    Moreover, it is assumed that given json formats include 'width' and 'height'.
    Function changes the structure of json files as: class, idx, isModified,
    xMin, yMin, xMax, yMax
    """

    image1 = io.imread(groundTruthImage)
    gtImageName, gtImgExtension = os.path.splitext(groundTruthImage)
    with open(gtImageName + '.json') as gtJsonFile:
         gtData = json.load(gtJsonFile)
    gtSamples = gtData['samples']

    image2 = io.imread(outputMaskImage)
    outputImageName, outputImgExtension = os.path.splitext(outputMaskImage)
    with open(outputImageName + '.json') as outputJsonFile:
         outputData = json.load(outputJsonFile)
    outputSamples = outputData['samples']

    struct = {
      "samples": [{
        "class" : {"type" : "string"},
        "idx": {"type": "string"},
        "isModified": {"type": "string"},
        "xMin": {"type": "string"},
        "yMin": {"type": "string"},
        "xMax" : {"type" : "string"},
        "yMax" : {"type" : "string"}
      }]
     }

    struct['samples'].clear()
    gtStructBoxes = {}
    gtStructBoxes['bbox'] = []
    outputStructBoxes = {}
    outputStructBoxes['bbox'] = []

    if image1.shape[0]!=image2.shape[0] or image1.shape[1]!=image2.shape[1]:
         for sample in gtSamples:
             classNum = str(sample['class'])
             idxNum = str(sample['idx'])
             modified = str(sample['isModified'])
             xMin = str(int(float(sample['x'])))
             yMin = str(int(float(sample['y'])))
             xMax = str((int(float(sample['x'])) + int(float(sample['width'])) - 1))
             yMax = str((int(float(sample['y'])) + int(float(sample['height'])) - 1))
             gtStruct = append_to_json(struct, classNum, idxNum, modified, xMin, yMin, xMax, yMax)
             gtStructBoxes['bbox'].append([int(xMin), int(yMin), int(xMax), int(yMax)])
         for sample in outputSamples:
             classNum = str(sample['class'])
             idxNum = str(sample['idx'])
             modified = str(sample['isModified'])
             xMin = str((image1.shape[1]/image2.shape[1])*int(float(sample['x'])))
             yMin = str((image1.shape[0]/image2.shape[0])*int(float(sample['y'])))
             xMax = str((image1.shape[1]/image2.shape[1])*(int(float(sample['x'])) + int(float(sample['width'])) - 1))
             yMax = str((image1.shape[0]/image2.shape[0])*(int(float(sample['y'])) + int(float(sample['height'])) - 1))
             outputStruct = append_to_json(struct, classNum, idxNum, modified, xMin, yMin, xMax, yMax)
             outputStructBoxes['bbox'].append([int(xMin), int(yMin), int(xMax), int(yMax)])
    else:
        for sample in gtSamples:
             classNum = str(sample['class'])
             idxNum = str(sample['idx'])
             modified = str(sample['isModified'])
             xMin = str(int(float(sample['x'])))
             yMin = str(int(float(sample['y'])))
             xMax = str((int(float(sample['x'])) + int(float(sample['width'])) - 1))
             yMax = str((int(float(sample['y'])) + int(float(sample['height'])) - 1))
             # outputStruct can be used later so it is held
             gtStruct = append_to_json(struct, classNum, idxNum, modified, xMin, yMin, xMax, yMax)
             gtStructBoxes['bbox'].append([int(xMin), int(yMin), int(xMax), int(yMax)])
        for sample in outputSamples:
             classNum = str(sample['class'])
             idx_num = str(sample['idx'])
             modified = str(sample['isModified'])
             xMin = str(int(float(sample['x'])))
             yMin = str(int(float(sample['y'])))
             xMax = str((int(float(sample['x'])) + int(float(sample['width'])) - 1))
             yMax = str((int(float(sample['y'])) + int(float(sample['height'])) - 1))
             # outputStruct can be used later so it is held
             outputStruct = append_to_json(struct, classNum, idx_num, modified, xMin, yMin, xMax, yMax)
             outputStructBoxes['bbox'].append([int(xMin), int(yMin), int(xMax), int(yMax)])
    return gtStructBoxes, outputStructBoxes, image1

def calculate_areas_of_bbox(gtBoxes):
     """
     A function that calculates a dictionary of bounding boxes'
     (each bbox has structure of [xMin,yMin,Xmax,yMax]) areas
     """
     bboxAreaList = []
     for i in range(len(gtBoxes['bbox'])):
          bboxAreaList.append((gtBoxes['bbox'][i][2]-gtBoxes['bbox'][i][0])*(gtBoxes['bbox'][i][3]-gtBoxes['bbox'][i][1]))
     return bboxAreaList


def calculate_iou(gtBbox, predBbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and
    return the IoU ratio.
    '''
    xTopLeftGt, yTopLeftGt, xBottomRightGt, yBottomRightGt = gtBbox
    xTopLeftPred, yTopLeftPred, xBottomRightPred, yBottomRightPred = predBbox

    if (xTopLeftGt > xBottomRightGt) or (yTopLeftGt> yBottomRightGt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (xTopLeftPred > xBottomRightPred) or (yTopLeftPred> yBottomRightPred):
        raise AssertionError("Predicted Bounding Box is not correct",xTopLeftPred, xBottomRightPred,yTopLeftPred,yBottomRightGt)

    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(xBottomRightGt< xTopLeftPred):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        return 0.0
    if(yBottomRightGt< yTopLeftPred):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        return 0.0
    if(xTopLeftGt> xBottomRightPred): # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        return 0.0
    if(yTopLeftGt> yBottomRightPred): # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        return 0.0

    gtBboxArea = (xBottomRightGt -  xTopLeftGt + 1) * (  yBottomRightGt -yTopLeftGt + 1)
    predBboxArea =(xBottomRightPred - xTopLeftPred + 1 ) * ( yBottomRightPred -yTopLeftPred + 1)

    x_top_left =np.max([xTopLeftGt, xTopLeftPred])
    y_top_left = np.max([yTopLeftGt, yTopLeftPred])
    x_bottom_right = np.min([xBottomRightGt, xBottomRightPred])
    y_bottom_right = np.min([yBottomRightGt, yBottomRightPred])

    intersectionArea = (x_bottom_right- x_top_left + 1) * (y_bottom_right-y_top_left  + 1)

    unionArea = (gtBboxArea + predBboxArea - intersectionArea)

    return intersectionArea/unionArea

def get_image_results(gtBoxes, predBoxes, iouThr, gtImage, bboxAreaList):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gtBoxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        predBoxes (dict): dict of dicts of 'boxes' (formatted like `gtBoxes`)
            and 'scores'
        iouThr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    allPredIndices= range(len(predBoxes))
    allGtIndices=range(len(gtBoxes))
    if len(allPredIndices)==0:
        tp=0
        fp=0
        fn=0
        return {'truePositive':tp, 'falsePositive':fp, 'falseNegative':fn}
    if len(allGtIndices)==0:
        tp=0
        fp=0
        fn=0
        return {'truePositive':tp, 'falsePositive':fp, 'falseNegative':fn}

    gtIdxThr = []
    predIdxThr = []
    ious = []
    for ipb, predBox in enumerate(predBoxes):
        for igb, gtBox in enumerate(gtBoxes):
            iou = calculate_iou(gtBox, predBox)
            if iou > iouThr:
                gtIdxThr.append(igb)
                predIdxThr.append(ipb)
                ious.append(iou)
    iouSort = np.argsort(ious)[::1]

    # Find the number of false negatives for small objects
    # Later this number will be subtracted from the number "fn"
    allSmallObjectsIou = {}
    indexList = []
    fnNumberOfSmallObjects = 0
    for i in range(len(gtBoxes)):
         if ((gtImage.shape[0]*gtImage.shape[1])/1000 > bboxAreaList[i]):
              indexList.append(i)
              smallObjectsIous = []
              for j in range(len(predBoxes)):
                   newIou = calculate_iou(gtBoxes[i], predBoxes[j])
                   smallObjectsIous.append(newIou)
                   allSmallObjectsIou['Small_' + str(i)] = smallObjectsIous
              if sum(allSmallObjectsIou['Small_' + str(i)]) == 0:
                 fnNumberOfSmallObjects = fnNumberOfSmallObjects+1

    if len(iouSort)==0:
        tp=0
        fp=0
        fn=0
        return {'truePositive':tp, 'falsePositive':fp, 'falseNegative':fn}
    else:
        gtMatchIdx=[]
        predMatchIdx=[]
        for idx in iouSort:
            gtIdx=gtIdxThr[idx]
            prIdx= predIdxThr[idx]
            # If the boxes are unmatched, add them to matches
            if(gtIdx not in gtMatchIdx) and (prIdx not in predMatchIdx):
                gtMatchIdx.append(gtIdx)
                predMatchIdx.append(prIdx)
        tp = len(gtMatchIdx)
        fp = len(predBoxes) - len(predMatchIdx)
        fn = len(gtBoxes) - len(gtMatchIdx) - fnNumberOfSmallObjects
    return {'truePositive': tp, 'falsePositive': fp, 'falseNegative': fn}

def bbox_perf_metrics(imgResults, performanceMetricType):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    truePositive=0
    falsePositive=0
    falseNegative=0
    for img_id, res in imgResults.items():
        truePositive +=res['truePositive']
        falsePositive += res['falsePositive']
        falseNegative += res['falseNegative']
        if performanceMetricType == 'oe':
             oe = falsePositive + falseNegative
             print('Overall Error = ', oe)
             return oe
        elif performanceMetricType == 'prec':
             try:
                 precision = truePositive/(truePositive + falsePositive)
                 print('Precision = ', precision)
                 return precision
             except ZeroDivisionError:
                 precision=0.0
        elif performanceMetricType == 'recall':
             try:
                 recall = truePositive/(truePositive + falseNegative)
                 print('Recall = ', recall)
                 return recall
             except ZeroDivisionError:
                 recall=0.0
        elif performanceMetricType == 'fmeas':
             try:
                 prec = truePositive/(truePositive + falsePositive)
                 recall = truePositive/(truePositive + falseNegative)
                 fmeas = 2 * prec * recall / (prec+recall)
                 print('F-Measure = ', fmeas)
                 return fmeas
             except ZeroDivisionError:
                 fmeas=0.0
        elif performanceMetricType == 'jc':
             try:
                  jc = truePositive / (truePositive+falsePositive+falseNegative)
                  print('Jaccard Coefficient = ', jc)
                  return jc
             except ZeroDivisionError:
                  jc=0.0
        else:
             print("Warning: Bbox-based supported metrics are 'oe', 'prec', 'recall', 'fmeas', 'jc'.")
             return None

# Main Function
def perf_eval(groundTruthImage, outputMaskImage, performanceMetricType, groundTruthType):
    """
    groundTruthImage: Ground truth image to test the accuracy of image analysis processes.
    outputMaskImage: Output image of the change detection algorithm.
    performanceMetricType:
       a) For the pixel-based case it can be:
         'pcc' -> percentage correct classification (accuracy)
         'oe' -> overall error
         'kc' -> Kappa coefficient
         'jc' -> Jaccard coefficient
         'yc' -> Yule coefficient
         'prec' -> precision
         'recall' -> recall
         'fmeas' -> F-measure
         'sp' -> specificity
         'fpr' -> false positive rate
         'fnr' -> false negative rate
         'pwc' -> percentage of wrong classifications
       b) For the bbox-based case it can be:
         'oe' -> overall error
         'prec' -> precision
         'recall' -> recall
         'fmeas' -> F-measure
         'jc' -> Jaccard coefficient
    groundTruthType: It can be 'pixel' or 'bbox'.
    """
    if groundTruthType == 'pixel':
         [image1Converted, image2Converted] = pixel_prepare_data(groundTruthImage, outputMaskImage)
         region_props, labelList, bboxList, areaList, bboxAreaList, convexImageList, coordinateList, imageList = get_object_properties(image1Converted)
         [tp, fp, tn, fn] = pixel_perf_measure(image1Converted, image2Converted, areaList, coordinateList)
         score = pixel_perf_metrics(tp, fp, tn, fn, performanceMetricType)
    elif groundTruthType == 'bbox':
         [gtStructBoxes, outputStructBoxes, image1] = bbox_prepare_data(groundTruthImage, outputMaskImage)
         bboxAreaList = calculate_areas_of_bbox(gtStructBoxes)
         imgResults = {}
         imgResults['confusion_matrix_elements'] = get_image_results(gtStructBoxes['bbox'], outputStructBoxes['bbox'], iouThr, image1, bboxAreaList)
         score = bbox_perf_metrics(imgResults, performanceMetricType)
    else:
         print("Warning: Supported ground truth types are 'pixel' and 'bbox'.")
    return score

# pixel images
#groundTruthImage = 'pixel_1.jpg'
#outputMaskImage = 'pixel_2.jpg'
# bbox images
groundTruthImage = 'bbox_1.bmp'
outputMaskImage = 'bbox_2.bmp'
iouThr = 0.5

performanceMetricType = 'recall'
groundTruthType =  'bbox'

score = perf_eval(groundTruthImage, outputMaskImage, performanceMetricType, groundTruthType)
