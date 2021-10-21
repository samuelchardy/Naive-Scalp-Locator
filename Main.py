import sys
import os.path as op
import time
from math import sqrt

import numpy as np
import nibabel as nib
from PIL import Image
import pyvista as pv
from mne import write_surface
from scipy.spatial import ConvexHull

from progress.bar import ChargingBar


def findNearest(data, value):
    xyDiff = np.abs(data-value)
    xyDiff = (xyDiff[:, 0]**2) + (xyDiff[:, 1]**2)
    xyDiff = np.array([sqrt(x) for x in xyDiff])
    return np.mean(xyDiff)



def makeImageBlackAndWhite(data):
    dataBW = data.copy()
    meanOfDat = np.mean(dataBW)
    dataBW[np.where(dataBW < meanOfDat)] = 0
    dataBW[np.where(dataBW > meanOfDat)] = 255
    return dataBW



def detectSide(data, dir, view, sliceNum):
    dataBW = makeImageBlackAndWhite(data)
    surfaceCoords = []

    dirVars = {"right": (0, 0, 256, 256, 1, 1),
                "left": (255, 255, -1, -1, -1, -1),
                "up":   (0, 255, 256, -1, 1, -1),
                "down": (255, 0, -1, 256, -1, 1)}

    start1, start2, end1, end2, inc1, inc2 = dirVars[dir]

    for i in range(start1, end1, inc1):
        reachedSurf = False
        for j in range(start2, end2, inc2):
            row = i
            col = j
            if dir == "up" or dir == "down":
                row = j
                col = i           
            if (dataBW[row, col, 0] > 140):
                if (reachedSurf is False):
                    data[row, col, :] = [255, 0, 0]
                    reachedSurf = True
                    if view == 0:
                        surfaceCoords.append(np.array([sliceNum, row, col]))
                    elif view == 1:
                        surfaceCoords.append(np.array([row, sliceNum, col]))
                    else:
                        surfaceCoords.append(np.array([row, col, sliceNum]))
                    break
    return data, np.array(surfaceCoords)



def makeSlices(data, invRes=1):
    slices0 = [data[x, :, :] for x in range(0, data.shape[0], invRes)]
    slices1 = [data[:, x, :] for x in range(0, data.shape[1], invRes)]
    slices2 = [data[:, :, x] for x in range(0, data.shape[2], invRes)]
    indices = [x for x in range(0, data.shape[0], invRes)]
    slices = [slices0, slices1, slices2]
    surfVals = np.array([-1, -1, -1])

    bar = ChargingBar("Detecting scalp edges", max=(len(slices0)*3))

    for j in range(len(slices)):
        sliceI = slices[j]
        for i in range(len(sliceI)):
            im = Image.fromarray(sliceI[i]).convert('RGB')
            dat = np.asarray(im)

            for direc in ["right", "left", "up", "down"]:
                dat, surfVal = detectSide(dat, direc, j, indices[i])

                if surfVal.shape[0] > 0:
                    surfVals = np.vstack((surfVals, surfVal))

            # im = Image.fromarray(np.uint8(dat))
            # im.save(f"{j}-{i}.png")
            bar.next()
    bar.finish()
    return surfVals



def outputSurface(data):

    surf = ConvexHull(data)

    # cloud = pv.PolyData(data)
    # surf = cloud.delaunay_3d()
    # #print(surf.cells.shape)

    # surf.plot(show_edges=True)
    # print(cloud.array_names)
    # print(surf.array_names)
    #print(surf.extract_all_edges().points)
    #edges = surf.cells.reshape((505294,3))
    # edges = surf.extract_all_edges().points
    #print(edges.shape)

    write_surface("outer_skin.surf", surf.points, surf.simplices, overwrite=True, file_format="freesurfer")



def outlierRemoval(data):
    bar = ChargingBar("Removing outliers", max=data.shape[0])
    distBetweenPoints = []

    for point in data:
        distBetweenPoints.append(findNearest(data, point))
        bar.next()

    meanDist = np.mean(distBetweenPoints)
    stdDist = np.std(distBetweenPoints)

    errorIndOver = np.where(distBetweenPoints > meanDist+(2*stdDist))
    errorIndUnder = np.where(distBetweenPoints < meanDist-(2*stdDist))
    errorInd = np.hstack((errorIndOver, errorIndUnder))

    data = np.delete(data, errorInd, 0)
    bar.finish()
    return data



def generateSurface(subjectID, invRes=1):
    plot = nib.load(f"../subjects/{subjectID}/mri/T1.mgz")
    imgData = plot.get_fdata()

    startTime = time.time()
    surfacePoints = makeSlices(imgData, invRes)
    surfacePoints = outlierRemoval(surfacePoints)
    surfacePoints[:,0] = surfacePoints[:,0]-124
    surfacePoints[:,1] = surfacePoints[:,1]-128
    surfacePoints[:,2] = surfacePoints[:,2]-128
    surfacePoints[:,1], surfacePoints[:,2] = surfacePoints[:,2], surfacePoints[:,1].copy()
    surfacePoints[:,2] = surfacePoints[:,2]*-1
    outputSurface(surfacePoints)
    print(f"Time: {time.time()-startTime}")






subjectID = sys.argv[1]
generateSurface(subjectID, invRes=25)






#im = Image.fromarray(np.uint8(dat))
#im.save(f"{i}.png")

# im = Image.fromarray(slice_0)
# im.save("images/test.png", "png")

# test = fig2img(plot)
# test.save(f"images/test.png")

# test = fig2data(plot)

# print(test)
# plot.savefig(f"images/test.png")