import numpy as np
from threading import Thread
import sklearn.feature_extraction.image as extraction

import skimage

from skimage import io
import matplotlib.pyplot as plt

import os
import csv

#Constants and dir names
csv_file        = "points_of_interest.csv"
patches_dir     = "patches3"
auto_patches    = "automated"

PATCH_SIZE      = 256

PATH            = 0
X               = 1
Y               = 2

THRESH_ABS      = 300.0
THRESH_NORM     = 0.001

def hand_pick_points(artist_path):
    """
    Automation for selecting points from an image by hand. Interacts through console and saves inputs into csv file
    :param artist_path: path of all the images to select from
    :return:
    """
    with open(os.path.join(artist_path, csv_file), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        points = []

        def record_csv():
            finished = False
            while not finished:
                x = input("enter X")
                y = input("enter Y")
                finished = input("finished? y/n") == 'y'
                if ('halt' in [x, y, finished]):
                    print ('halted')
                    return
                try:
                    if int(x) > 0 and int(y) > 0:
                        points.append((x,y))
                except:
                    pass


        for im_name in os.listdir(artist_path):
            im_file = os.path.join(artist_path, im_name)

            try:
                im = io.imread(im_file)
                print(im_file)
                Thread(target=record_csv).start()
                plt.imshow(im)
                plt.show()
                for point in points:
                    writer.writerow([im_file, point[X-1], point[Y-1]])
                points = []
            except:
                pass


def save_patches_from_csv(artist_path):
    """
    Used after hand picking patches. Saves them to folder.
    :param artist_path:
    :return:
    """
    try:
        os.mkdir(os.path.join(artist_path, patches_dir))
    except:
        pass
    if os.path.isfile(os.path.join(artist_path, csv_file)):

        with open(os.path.join(artist_path, csv_file)) as points_of_interest:
            desc_reader = csv.reader(points_of_interest, delimiter=",")
            im_path = ""
            for row in desc_reader:
                if not row or row[0] == "":
                    break
                if row[PATH] != im_path:
                    im_path = row[PATH]
                    im = io.imread(im_path)
                    # im =

                x = int(row[X])
                y = int(row[Y])
                dim = PATCH_SIZE / 2
                patch = im[int(max(0, y-dim)): int(min(y+dim, im.shape[0])), int(max(0, x-dim)) : int(min(x+dim, im.shape[1])), ...]

                patch = patch

                plt.imshow(patch)
                plt.show()
                patch_name = os.path.split(im_path)[1][:-4] + "," + row[X] + "," + row[Y]

                io.imsave(os.path.join(artist_path, patches_dir, patch_name) + ".png", patch)
    else:
        print("No point csv file found for this artist (under {})".format(artist_path))


def filter_image_file(file):
    return file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".JPG")

def automate_patches(artist_path, patches_lim=10):
    """
    Performs patch making automatically for given image library
    :param artist_path:
    :return:
    """
    patch_window = (PATCH_SIZE, PATCH_SIZE)
    auto_dir = os.path.join(artist_path, patches_dir, auto_patches)
    try:
        try:
            os.mkdir(os.path.join(artist_path, patches_dir))
        except:
            pass
        os.mkdir(auto_dir)
    except Exception as e:
        pass
    for im_name in filter(filter_image_file, os.listdir(artist_path)):
        im_file = os.path.join(artist_path, im_name)

        try:
            print("Extracting patches from {}".format(im_file))
            filtered = 0
            saved = 0
            im = io.imread(im_file)
            finished = False
            i = 1
            while not finished:
                patches = extraction.extract_patches_2d(image=im, patch_size=patch_window, max_patches=patches_lim)
                for i, patch in enumerate(patches):
                    # Perfrom threshold analysis, and check if need to generate more patches.
                    v1 = np.var(patch)
                    v2 = np.var(skimage.color.rgb2gray(patch))
                    if not (v1 < THRESH_ABS or v2 < THRESH_NORM ):
                        io.imsave(os.path.join(auto_dir, im_name + str(i) + ".png"), patch)
                        saved += 1
                        if saved == 10:
                            break
                    else :
                        filtered += 1
                else:
                    print(" -- Filtered out {}".format(filtered))
                finished = saved >= 10
                i += 1

        except:
            print("Could not extract for {}".format(im_file))
            continue
            pass


if __name__ == '__main__':

    hand_pick_points("artist_data/Pierre-Auguste Renoir/paint")
    save_patches_from_csv("artist_data/Pierre-Auguste Renoir/paint")


