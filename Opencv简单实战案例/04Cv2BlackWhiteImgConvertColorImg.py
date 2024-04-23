import numpy as np
import cv2 as cv

# name of the input file
imname = 'Images/DogFace.jpg'

# read in the image
im = cv.imread(imname)
im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
print(im.shape)
cv.imshow("source image", im)

# convert to double (might want to do this later on to save memory)
# im = img_as_float(im)

# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(int)

# separate color channels
b = im[:height]
g = im[height: 2 * height]
r = im[2 * height: 3 * height]

# align the images
# functions that might be useful for aligning the images include: np.roll, np.sum
# ag = align(g, b)
# ar = align(r, b)

# create a color image
# im_out = cv.merge((ar, ag, b))  # this line should be activated after implementing the align functions
im_out = cv.merge((b, g, r))  # this line should be deleted after implementing the align functions

# save and display the output image
cv.imwrite("out/out_fname.jpg", im_out)
cv.imshow("output image", im_out)

cv.waitKey(0)