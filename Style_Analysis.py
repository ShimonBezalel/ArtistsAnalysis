import numpy as np
import os

import Artist_Descriptors
import Patch
import Map_Artists

from enum import Enum


ARTIST_LIB = "artist_data"
PAINT_DIR = "paint"

class DescStradegy(Enum):
    CANNY       = 'canny'
    CANNY2      = 'canny2'
    NORMALIZED  = 'none'
    FREQUENCY   = 'fft'
    COLOR_HIST  = 'hues'
    NAME        = 'names'


class Analyzer:

    def __init__(self):
        self.num_of_artists = 0
        self.artist_map = np.zeros(shape=(0, 4))
        self.artist_means = np.zeros(shape=(0, 1))

    def train(self,
              library_path=ARTIST_LIB,
              artist_filter=lambda x: "DS_Store" not in x,
              strategy=DescStradegy.CANNY.value):

        print("""Describing all artists in {} using:\n 
        Stradegy: {}\t
              """.format(library_path, strategy))
        for artist in filter(artist_filter, os.listdir(library_path)):
            print("Describing: {}".format(artist))
            rel_path = os.path.join(ARTIST_LIB, artist, PAINT_DIR)
            # Patch.automate_patches(rel_path)
            Artist_Descriptors.describe_artist(rel_path,
                                               canny=(strategy == DescStradegy.CANNY.value),
                                               canny2=(strategy == DescStradegy.CANNY2.value),
                                               frequency=(strategy == DescStradegy.FREQUENCY.value),
                                               color_hist=(strategy == DescStradegy.COLOR_HIST.value),
                                               names=(strategy == DescStradegy.NAME.value)
                                               )
            self.num_of_artists += 1


if __name__ == '__main__':
    analysor = Analyzer()
    analysor.train(strategy='names', artist_filter=lambda name: "DS_Store" not in name )
