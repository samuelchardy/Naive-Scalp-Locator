# Naive-Scalp-Locator
Maps the scalp from T1 mri scans and exports this as a surface.

## Select Image
![Alt text](examples/example.png?raw=true "T1 Slice")

T1 scan data is volumetric, meaning that the data is a set of 3d coordinates, in which each cell is a pixel (voxel) of the scan. By looking at multiple slices from each dimension we can develop an idea of this 3d volume through 2d images.

## Pre-process Image
![Alt text](examples/example2.png?raw=true "T1 Slice")

As each pixel of the image stores RGB values for colour, it is easier if we unify the colour of each pixel to either black or white. This means colour can be determined via a single RGB value as the others are all identical.

## Detect Edges
![Alt text](examples/example3.png?raw=true "T1 Slice")
![Alt text](examples/example4.png?raw=true "T1 Slice")

For each row go across all columns and find the first white pixel. To achieve the above the same process of finding the first white pixel must be replicated on the image data by traversing the matrix as if moving up, down, left, and right across the image. With the colour settings back to normal the surface can be seen clearer.

## Outlier Removal
![Alt text](examples/example5.gif?raw=true "T1 Slice")
![Alt text](examples/example6.gif?raw=true "T1 Slice")

As a placeholder for a more effective method of outlier detection we currently calculate the distance between each point and every other point. Doing this allows us to calculate the mean and standard deviation of distances between all points, from these we remove points which have a z-score greater than 2 (are outside 2 standard deviations from the mean). The first gif shows pre-removal points, and the second gif shows post-removal points.
