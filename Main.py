import sys
import os.path as op
from math import sqrt

import numpy as np
import nibabel as nib

from mne.viz.misc import get_subjects_dir
from mne.viz.misc import _check_mri
from mne.viz.misc import _get_bem_plotting_surfaces
from mne.viz.misc import _plot_mri_contours

from PIL import Image

import matplotlib.pyplot as plt




def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis = 2)
    return buf



def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w ,h), buf.tobytes())


# def detectSide(data, dir):

#     dirVars = {"right": (0, 0, 256, 256, 1, 1),
#                 "left": (255, 255, -1, -1, -1, -1),
#                 "up":   (0, 255, 256, -1, 1, -1),
#                 "down": (255, 0, -1, 256, -1, 1)}

#     start1, start2, end1, end2, inc1, inc2 = dirVars[dir]

#     for i in range(start1, end1, inc1):
#         reachedSurf = False

#         for j in range(start2, end2, inc2):
#             row = i
#             col = j

#             if dir == "up" or dir == "down":
#                 row = j
#                 col = i
            
#             if (data[row, col, 0] > 10) and (reachedSurf is False):
#                 data[row, col, :] = [255, 0, 0]
#                 reachedSurf = True

#     return data


def findNearest(data, value):
    xyDiff = np.abs(data-value)
    print(xyDiff[0])
    xyDiff = (xyDiff[:, 0]**2) + (xyDiff[:, 1]**2)
    xyDiff = np.array([sqrt(x) for x in xyDiff])
    print(xyDiff[0])
    minDist = xyDiff.argmin()
    print(xyDiff[minDist])



def makeImageBlackAndWhite(data):
    dataBW = data.copy()
    meanOfDat = np.mean(dataBW)
    dataBW[np.where(dataBW < meanOfDat)] = 0
    dataBW[np.where(dataBW > meanOfDat)] = 255
    return dataBW



def detectSide(data, dir):
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
            
            if (dataBW[row, col, 0] > 10):
                if (reachedSurf is False):
                    data[row, col, :] = [255, 0, 0]
                    reachedSurf = True
                    surfaceCoords.append(np.array([row, col]))

    return data, np.array(surfaceCoords)




subjectID = sys.argv[1]

plot = nib.load(f"../subjects/{subjectID}/mri/T1.mgz")
epi_img_data = plot.get_fdata()

# slices = [epi_img_data[x, :, :] for x in range(50, 200, 10)]

slices = [epi_img_data[:, x, :] for x in range(50, 200, 10)]
# slice_2 = epi_img_data[:, :, 100]

# slices = [slice_0, slice_1, slice_2]
surfVals = None

for i in range(len(slices)):
    im = Image.fromarray(slices[i]).convert('RGB')
    dat = np.asarray(im)

    surf = np.zeros((256, 256, 256))

    for direc in ["right", "left", "up", "down"]:
        dat, surfVals = detectSide(dat, direc)
        #print(surfVals.shape)
        dat[surfVals[:, 0], surfVals[:, 1], :] = [0, 0, 255]

    im = Image.fromarray(np.uint8(dat))
    im.save(f"{i}.png")


# x = findNearest(surfVals, np.array([7,7]))






# im = Image.fromarray(slice_0)
# im.save("images/test.png", "png")

# test = fig2img(plot)
# test.save(f"images/test.png")


# test = fig2data(plot)



# print(test)
# plot.savefig(f"images/test.png")