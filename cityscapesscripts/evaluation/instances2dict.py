#!/usr/bin/python
#
# Convert instances from png files to a dictionary
#

from __future__ import print_function, absolute_import, division
import os, sys

# Cityscapes imports
from cityscapesscripts.evaluation.instance import *
from cityscapesscripts.helpers.csHelpers import *

import zarr


def instances2dict(imageFileList, verbose=False, key=None):
    imgCount     = 0
    instanceDict = {}

    if not isinstance(imageFileList, list):
        imageFileList = [imageFileList]

    if verbose:
        print("Processing {} images...".format(len(imageFileList)))

    for imageFileName in imageFileList:
        # Load image

        if os.path.splitext(imageFileName)[-1] == ".zarr":
            img = np.array(zarr.open(imageFileName, 'r')[key])
        else:
            img = Image.open(imageFileName)

        # Image as numpy array
        imgNp = np.array(img)

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        if key is None:
            for instanceId in np.unique(imgNp):
                instanceObj = Instance(imgNp, instanceId)

                instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())
        elif key == 'volumes/gt_instances':
            for instanceId in np.unique(imgNp):
                if instanceId == 0:
                    continue
                instanceObj = Instance(imgNp, instanceId)

                instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())
        elif key == 'volumes/gt_labels':
            for idx in range(imgNp.shape[0]):
                instanceId = idx + 1
                img = imgNp[idx]*instanceId
                instanceObj = Instance(img, instanceId)

                instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())

        imgKey = os.path.abspath(imageFileName)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=' ')
            sys.stdout.flush()

    if verbose:
        print("")

    return instanceDict

def main(argv):
    fileList = []
    if (len(argv) > 2):
        for arg in argv:
            if ("png" in arg):
                fileList.append(arg)
    instances2dict(fileList, True)

if __name__ == "__main__":
    main(sys.argv[1:])
