"""
Catalogue Steps:
1. load in catalogue
2. split into healpixels, only include top N brightest stars in each healpixel
3. itterate through each healpixel create quads between X and Y in scale length. Scale length is determined using the angular seperation of the two furthest stars.
4. for each quad, convert into a hashcode
OUTPUT: cat_data (dataframe containing all stars in catalogue), cat_codes (4D kd tree containing all hashcodes), cat_quads (list of quads in catalogue, in data.index format)
"""

from astropy_healpix import HEALPix, healpy
from astropy.io import fits
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import itertools
import pickle as pkl

import utils as utils

def cat2codes(RA_lims, DE_lims, N):
    # loading in the data
    northern_data = fits.open('T2_northern.fit', ignore_missing_simple=True)
    southern_data = fits.open('T2_southern.fit', ignore_missing_simple=True)

    # converting data to pandas dataframe
    northern_df = pd.DataFrame(northern_data[1].data)
    southern_df = pd.DataFrame(southern_data[1].data)

    # concatenating dataframes
    cat_data = pd.concat([northern_df,southern_df],ignore_index=True)

    # resetting index
    cat_data = cat_data.reset_index(drop=True)
    cat_data = cat_data.copy()

    # create a list of healpix pixels for each coordinate in cat_data
    cat_data['healpix'] = healpy.ang2pix(128, cat_data['_RAJ2000'], cat_data['_DEJ2000'], nest=True, lonlat=True)

    # dropping the recno column
    cat_data = cat_data.drop('recno', axis=1)

    cat_data = cat_data.copy()
    # filter data
    cat_data = cat_data[cat_data['_RAJ2000'] >= RA_lims[0]]
    cat_data = cat_data[cat_data['_RAJ2000'] <= RA_lims[1]]
    cat_data = cat_data[cat_data['_DEJ2000'] >= DE_lims[0]]
    cat_data = cat_data[cat_data['_DEJ2000'] <= DE_lims[1]]
    print('Length of filtered data: ', len(cat_data))

    cat_data['count'] = 0
    healpix_count = pd.DataFrame({'healpix': cat_data['healpix'].unique()})
    healpix_count['count'] = 0

    # for each healpix pixel, only keep the N brightest stars
    for healpix in cat_data['healpix'].unique():
        healpix_data = cat_data[cat_data['healpix'] == healpix]
        healpix_data = healpix_data.sort_values(by=['VTmag']).head(N)
        healpix_data = healpix_data.iloc[:N]
        cat_data = cat_data.drop(cat_data[cat_data['healpix'] == healpix].index)
        cat_data = pd.concat([cat_data, healpix_data], ignore_index=True)

    # convert each point to a cartesian coordinate on a unit sphere
    cat_data['x'] = np.cos(np.radians(cat_data['_DEJ2000'])) * np.cos(np.radians(cat_data['_RAJ2000']))
    cat_data['y'] = np.cos(np.radians(cat_data['_DEJ2000'])) * np.sin(np.radians(cat_data['_RAJ2000']))
    cat_data['z'] = np.sin(np.radians(cat_data['_DEJ2000']))

    # relabel columns to 'RA' and 'DE
    cat_data = cat_data.rename(columns={'_RAJ2000': 'RA', '_DEJ2000': 'DE'})

    # set any HIP numbers that are -2147483648 to NaN
    cat_data['HIP'] = cat_data['HIP'].replace(-2147483648, np.nan)

    # sort cat_data by healpix pixel
    cat_data = cat_data.sort_values(by=['healpix', 'VTmag'])

    # reset index
    cat_data = cat_data.reset_index(drop=True)

    
    # create quads and hashcodes lists
    quads = []
    hashcodes = []

    # create a cKDTRee object for the x, y, z coordinates
    tree = cKDTree(cat_data[['x', 'y', 'z']])

    # setting up Dmin and Dmax
    Dmin = 2 * np.sin(np.radians(0.25)/2)
    Dmax = 2 * np.sin(np.radians(0.35)/2)

    print ('Dmin: ', Dmin)
    print ('Dmax: ', Dmax)



    # run the pass three times
    for i in range(10):
        run_pass(cat_data, tree, Dmax, Dmin, quads, hashcodes)
        print(' Number of quads: ', len(quads))

    # save quads and hashcodes as lists
    np.save('quads.npy', quads)
    np.save('hashcodes.npy', hashcodes)

    # save cat_data as a DataFrame
    with open('cat_data.pkl', 'wb') as f:
        pkl.dump(cat_data, f)

    




    
def run_pass(cat_data, tree, Dmax, Dmin, quads, hashcodes):
    # creating quads and hashcodes
    for healpix in cat_data['healpix'].unique():
        #ending the loop if there are 100 quads

        print('healpix: ', healpix)
        found = False
        # creating a list of all stars in the healpix pixel
        stars = cat_data[cat_data['healpix'] == healpix].index.tolist()
        
        # itterate through each star in stars
        for star in stars:
            if found == True:
                break
            # find all stars within Dmax of star
            neighbours = tree.query_ball_point(cat_data.loc[star, ['x', 'y', 'z']], Dmax)
            # drop star from neighbours
            neighbours.remove(star)

            # ensure that there are at least 3 neighbours
            if len(neighbours) < 3:
                continue
            
            # create a combination of 3 stars from neighbours
            for combination in itertools.combinations(neighbours, 3):
                # create a quad from the combination
                quad = [star, combination[0], combination[1], combination[2]]

                # make quad into a sorted tuple
                quad = tuple(sorted(quad))

                # check if quad is already in quads
                if quad in quads:
                    #print('quad already in quads')
                    continue


                inp = cat_data.loc[quad, ['x', 'y', 'z']].values

                _,_,scale = utils.findAB(inp)

                if (scale < Dmin) or (scale > Dmax):
                    #print('scale not in range')
                    continue

                # check which healpixel the centroid of the quad is in
                centroid = cat_data.loc[quad, ['RA','DE']].mean()
                centroid_healpix = healpy.ang2pix(128, centroid['RA'], centroid['DE'], nest=True, lonlat=True)

                # if the centroid is not in the same healpixel as the quad, continue
                if centroid_healpix != healpix:
                    #print('centroid not in same healpix as quad')
                    continue

                # add quad to quads
                quads.append(quad)
                # create hashcode for quad
                hashcode = tuple(utils.hashcode(cat_data.loc[quad, ['RA', 'DE']].values))
                hashcodes.append(hashcode)
                found = True
                break

        
            

                






# running code
cat2codes([9,14],[9,14],10)
#cat2codes([35,60],[50,60],10)

