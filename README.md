# Naive-Scalp-Locator
Maps the scalp from T1 mri scans and exports this as a surface.

## Methodology
### Split T1 scan volume data into multiple image slices like the one below.
![Alt text](examples/example.png?raw=true "T1 Slice")

T1 scan data is volumetric, meaning that the data is a set of 3d coordinates, in which each cell is a pixel (voxel) of the scan. By looking at multiple slices from each dimension we can develop an idea of this 3d volume through 2d images.

### Increase the contrast of the image.
![Alt text](examples/example2.png?raw=true "T1 Slice")

As each pixel of the image stores RGB values for colour, it is easier if we unify the colour of each pixel to either black or white. This means colour can be determined via a single RGB value as the others are all identical.

### For each go across all columns and find the first white pixel.
![Alt text](examples/example3.png?raw=true "T1 Slice")

To achieve the above the same process of finding the first white pixel must be replicated on the image data by traversing the matrix as if moving up, down, left, and right across the image. With the colour settings back to normal the surface can be seen clearer.
![Alt text](examples/example4.png?raw=true "T1 Slice")
