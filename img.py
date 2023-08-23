# image processing steps:
# 1. load in image
# 2. preprocess
# 3. find conrers in each quadrent
# 4. itterate through to create quads
# 5. convert each quad into a hashcode
# OUTPUT: img_data (dataframe containing all stars in image), img_codes (4D kd tree containing all hashcodes), img_quads (list of quads in image, in data.index format)

# imports
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import cKDTree
from itertools import combinations
import utils


# loading in the image
def img2codes(file):

    """ loading steps """
    # load the file using fits
    hdul = fits.open(file)

    # extract the data
    image = hdul[0].data

    # extract the header
    header = hdul[0].header

    """ producing target """

    # Create the WCS object and handle any exceptions
    try:
        t = WCS(header, naxis=2)
        # Get the image dimensions from the WCS
        try:
            image_width, image_height = t.pixel_shape
        except:
            print('fits WCS has {} axes, expected 2. Can accept 3'.format(t.naxis))
            _, image_width, image_height = t.array_shape
        # Calculate the center pixel coordinates
        center_x = image_width / 2.0
        center_y = image_height / 2.0

        center_coords = t.pixel_to_world(center_x, center_y)

        # Access the RA and Dec values directly from the SkyCoord object
        center_ra = center_coords.ra.deg
        center_dec = center_coords.dec.deg
        target = (float(center_ra), float(center_dec))
        print('target: ', target)

    except Exception as e:
        # Handle any exceptions that may occur during the WCS creation or conversion
        raise Exception("Failed to create WCS object: " + str(e))


    """ processing steps """

   # converting image into an 8-bit unsigned integer with values between 0 and 255
    image = image.astype(np.uint8)
    image = 255-image
    if len(np.shape(image)) == 3:
        # convert to greyscale not using cv2
        image = image[0,:,:]
    # apply a gaussian blur to the image
    image = cv2.GaussianBlur(image, (11, 11), 0)
    # creating a median smoothed image
    image_median = cv2.medianBlur(image, 31)
    # subtracting the median smoothed image from the original image
    image = cv2.subtract(image_median, image)
    image = 255-image

    """ insert variance thresholding here 
    
    
    """

    """ setting up detector """
    image_size = min([image_height, image_width])
    # Step 3: find the corners of the image
    params = cv2.SimpleBlobDetector_Params()
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 80
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.5
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector.create(params)

    """  finding the corners """
    # Detect blobs.
    keypoints = detector.detect(image)
    corners = [[keypoint.pt[0], keypoint.pt[1]] for keypoint in keypoints]
    sizes = [keypoint.size for keypoint in keypoints]

    coords = sorted(zip(sizes, corners), reverse=True)[:50]
    sizes, corners = map(list, zip(*coords))

    # creating img_data dataframe containing all stars in image cols = ['x', 'y', 'size']
    img_data = pd.DataFrame(corners, columns=['x', 'y'])    
    img_data['size'] = sizes
    img_data = img_data.sort_values(by=['size'], ascending=False)
    img_data['count'] = 0


    # creating img_quads list of quads in image, in data.index format
    tree = cKDTree(img_data[['x', 'y']].values)
    img_quads = []
    img_codes = []
    run_img_pass(img_data, image_size, tree, 10, img_quads, img_codes)

    return img_data, img_quads, img_codes, np.shape(image)



def run_img_pass(img_data, image_size, tree, cnt, img_quads, img_codes):
    # itterating through the corners to create quads
    for star in img_data.index.to_list():
        if img_data.loc[star, 'count'] >= cnt:
            continue
        # find the neighbouring stars within 0.35*image_size
        neighbours = tree.query_ball_point(img_data.loc[star, ['x', 'y']].values, 0.35*image_size)
        # remove the star itself from the neighbours
        neighbours.remove(star)
        # itterate through the neighbours to create quads
        for combination in combinations(neighbours, 3):
            quad = [star,*combination]

            # check if any of the stars in the quad have already been used 3 times
            if img_data.loc[quad, 'count'].max() >= cnt:
                continue
            # sort the quad
            quad = tuple(sorted(quad))
            # check if the quad is already in the list
            if quad in img_quads:
                continue # if it is, skip to the next combination

            # check the scale of the quad
            _,_,scale = utils.findAB(img_data.loc[quad, ['x', 'y']].values)
            # if the scale is not between 0.25 and 0.35, skip to the next combination
            if (scale < 0.25 * image_size) or (scale > 0.35 * image_size):
                continue

            # add the quad to the list
            img_quads.append(quad)
            # convert the quad into a hashcode
            img_codes.append(utils.hashcode(img_data.loc[quad, ['x', 'y']].values))

            # add 1 to the count of each star in the quad
            img_data.loc[list(quad), 'count'] += 1


















