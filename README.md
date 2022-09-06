# computerVision-performance-metrics
Calculating Performances with respect to Ground Truth Image

In this repository, you will find Python script for calculating performance metrics by using ground truth image and output image of a prediction. There are two types of ground truth type such as 'pixel' and 'bounding box'. If the 'bounding box' type is selected, then the format of the ground truth type is json as:

      "samples": [{
        "class" : {"type" : "string"},
        "idx": {"type": "string"},
        "isModified": {"type": "string"},
        "xMin": {"type": "string"},
        "yMin": {"type": "string"},
        "xMax" : {"type" : "string"},
        "yMax" : {"type" : "string"}
