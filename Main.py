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
    skullCol = findSkullColour(data)
    data[np.where(data < skullCol)] = 0
    data[np.where(data > skullCol)] = 255
    return data



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
            if (data[row, col, 0] > 140):
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

            im = Image.fromarray(np.uint8(dat))
            im.save(f"{j}-{i}.png")
            bar.next()

    surfVals = np.delete(surfVals, [0], 0)
    bar.finish()
    return surfVals



def outputSurface(data):
    surf = ConvexHull(data)
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



def findSkullColour(data):
    slice0Mid = np.mean(data[120, :, :])
    slice1Mid = np.mean(data[:, 120, :])

    col = np.mean([slice0Mid, slice1Mid])
    return col


def generateSurface(subjectID, invRes=1):
    plot = nib.load(f"../subjects/{subjectID}/mri/T1.mgz")
    imgData = plot.get_fdata()

    imgData = makeImageBlackAndWhite(imgData, )
    startTime = time.time()
    surfacePoints = makeSlices(imgData, invRes)
    surfacePoints = outlierRemoval(surfacePoints)
    surfacePoints[:,0] = surfacePoints[:,0]-124
    surfacePoints[:,1] = surfacePoints[:,1]-128
    surfacePoints[:,2] = surfacePoints[:,2]-128
    surfacePoints[:,1], surfacePoints[:,2] = surfacePoints[:,2], surfacePoints[:,1].copy()
    surfacePoints[:,2] = surfacePoints[:,2]*-1
    pl = pv.Plotter()
    _ = pl.add_points(surfacePoints, render_points_as_spheres=True, color='w', point_size=10)
    pl.show()
    outputSurface(surfacePoints)
    print(f"Time: {time.time()-startTime}")






subjectID = sys.argv[1]
generateSurface(subjectID, invRes=15)



