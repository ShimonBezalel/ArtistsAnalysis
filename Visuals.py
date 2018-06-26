"""
Visualization for "Who do I paint Like?" Project
This module should only be run after a full analysis was performed on data.

Author: Shimon Heimowitz
26.6.18
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
import time
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d
from skimage import io
import os
from sklearn.neighbors.nearest_centroid import NearestCentroid

from matplotlib.offsetbox import *

movements = {
        "Personal Portfolio": ([
                                ('Personal Portfolio 08 - 10', "#ad1818"),
                                ('Personal Portfolio 12 - 15', "#d60a0a"),
                                ('Personal Portfolio 16 - 18', "#ef3939")]
                               , '#470915'),
        "Impressionism (& Post)": ([('Edgar Degas', "#20ad16"),
                                    ('Pierre-Auguste Renoir', "#45b53d"),
                                    ('Henri Matisse', "#67a563"),
                                    ('Van Gogh', "#6d8e6b"),
                                    ('Claude Monet', "#a6eda8"),
                                    ('Edvard Munch', "#0ef214"),
                                    ('Paul Cezanne', "#c4efc5")]
                                   , '#053a06'),
        "Boroque & Dutch Golden Age": ([("Rembrandt van Rijn", "#bc9d67"),
                                        ("Peter Paul Rubens", "#897758"),
                                        ("Johannes Vermeer", "#efad3b")]
                                       , "#44300c"),
        "Pop": ([('Andy Warhol', "#772975"),
                 ('Keith Haring', "#d739db"),
                 ("Roy Lichtenstein", "#c477c6")]
                , "#2f0330"),
        "Surrealism": ([('Salvador Dali', "#831fc4"),
                        ('Frida Kahlo', "#ba74e8"),
                        ("Wolfgang Lettl", "#855ba0")]
                       , "#320a4c"),
        "Abstract": ([("Wassily Kandinsky", "#ede94e"),
                      ("Kazimir Malevich", "#d8cd79"),
                      ("Timothy Hawkesworth", "#d8d313"),
                      ("Fahrelnissa Zeid", "#dbd98a"),
                      ("Jackson Pollock", "#f2f21d")]
                     , "#56540e"),
        "Cubism": ([("Pablo Picasso", "#0359c1"),
                    ("Franz Marc", "#5fa3fc")]
                   , "#182d56")
    }

color_groups = movements
all_artists = []
for m in movements.values():
    for art, _ in m[0]:
        all_artists.append(art)


def is_portfolio(targets):
    # return  "Personal Portfolio 16 - 18" == targets
    return np.bitwise_or(np.bitwise_or("Personal Portfolio 08 - 10" == targets,
                                       "Personal Portfolio 12 - 15" == targets),
                                        "Personal Portfolio 16 - 18" == targets)

def parse_path(path):
    artist_path, im_name = path.split("/patches3/automated/")
    return os.path.join( artist_path, im_name[:-5])

def training_scatter():
    """
    Visual the data of all the training artists used to create the original mapping.
    """
    ALPHA_LOW = 0.02
    ALPHA_SEMI = 0.3
    ALPHA_HIGH = 1-ALPHA_LOW

    fig = plt.figure(figsize=(10, 6), tight_layout=False)
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.autoscale()
    medians = dict()

    leg = []
    for group in color_groups:
        # if "Portfolio" not in group:
        #     continue
        for artist, color in movements[group][0]:
            leg.append(artist)
            indices = np.where(targets == artist)[0]
            group_data = data[indices]
            x = np.median(group_data[..., 0])
            y = np.median(group_data[..., 1])
            z = np.median(group_data[..., 2])
            medians[artist] = np.array((x, y, z))
            ax.scatter(x, y, z,
                       edgecolors=movements[group][1],
                       facecolors=color,
                       lw=0.5,
                       alpha=0.9,
                       s=200
                       )
    ax.legend(labels=leg, loc='upper right', bbox_to_anchor=(1, 1), ncol=1,
              bbox_transform=plt.gcf().transFigure)

    gen_scatters = []
    for group in color_groups:
        a = ALPHA_LOW if "Portfolio" not in group else ALPHA_SEMI
        # if "Portfolio" not in group:
        #     continue
        for artist, color in movements[group][0]:
            leg.append(artist)
            indices = np.where(targets == artist)[0]
            group_data = data[indices]
            gen_scatters.append(ax.scatter(group_data[..., 0],
                                           group_data[..., 1],
                                           group_data[..., 2],
                                           edgecolors=movements[group][1],
                                           facecolors=color,
                                           lw=0.5,
                                           alpha=a,
                                           s=20
                                           ))

    def swap(event):
        if event.dblclick:
            for sc in gen_scatters:
                sc.set_alpha(np.abs(1 - sc.get_alpha()))
        elif event.inaxes == ax:
            for sc in gen_scatters:
                cont, ind = sc.contains(event)
                if cont:
                    choice = np.random.choice(ind['ind'])
                    sc.set_alpha(1 - ALPHA_LOW)
                    for other_sc in filter(lambda s: s != sc, gen_scatters):
                        other_sc.set_alpha(ALPHA_LOW)
                    break
            else:
                pass
                for sc in gen_scatters:
                    sc.set_alpha(ALPHA_LOW)

        fig.canvas.draw_idle()

    connection_id = fig.canvas.mpl_connect('button_press_event', swap)

    plt.show()

def Centroid():
    """
    Visualize the new data (Personal Portfolio) on the mapping of all the artists, by coloring the new data points with
    the artist's colors. The centroid decides for each data point which artist is fits to, and then a tally is performed
    on a per-image basis.
    """

    # Create new Figure and an Axes which fills it.
    fig = plt.figure(figsize=(12, 6), tight_layout=True)
    # ax = p3.Axes3D(fig)
    ax = fig.add_subplot(122, projection='3d')
    ax_im = fig.add_subplot(221)
    ax_legend = fig.add_subplot(223)
    ax_legend.imshow(plt.imread("presentation/pie legend.png"))

    ax_im.grid(False)
    ax_legend.grid(False)
    ax_im.set_xticks([])
    ax_im.set_yticks([])
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel('Hue Component 1', fontsize=15)
    ax.set_ylabel('Hue Component 2', fontsize=15)
    ax.set_zlabel('Canny Component', fontsize=15)
    ax.set_title('3 Component PCA - Hues & Canny', fontsize=20)

    my_indices = np.where(is_portfolio(targets))[0]

    clf = NearestCentroid()
    training_data = data[np.where(np.bitwise_not(is_portfolio(targets)))[0]]
    labels = targets[np.where(np.bitwise_not(is_portfolio(targets)))[0]]

    test_data = data[my_indices]

    clf.fit(training_data, labels)

    im_paths = src_paths[my_indices]
    im_coords = data[my_indices]
    ind = np.argsort(im_paths)
    s_im_paths = im_paths[ind]
    s_im_coords = im_coords[ind]

    unique_im_paths, unique_indices = np.unique(list(map(parse_path, s_im_paths)), return_index=True)
    dest_coords = s_im_coords[unique_indices]

    print("Calculating Centroids")

    res = clf.predict(test_data[ind])

    im_desc_per_artist = {artist : ([], []) for artist in all_artists}
    for i, index in enumerate(unique_indices[:-1]):
        r = res[index : unique_indices[i+1]]
        median = np.median(s_im_coords[index: unique_indices[i+1]], axis=0)
        n, c = np.unique(r, return_counts=True)
        t = np.argmax(c)
        im_desc_per_artist[n[t]][0].append(median)
        im_desc_per_artist[n[t]][1].append(unique_im_paths[i])

    print("Graphing test data")

    image_dict = {u: plt.imread(u) for u in unique_im_paths}
    im = ax_im.imshow(image_dict[unique_im_paths[0]])

    leg = []
    gen_scatters = []

    for movement in movements.values():
        for artist, color in movement[0]:
            leg += [artist]
            result_indices = np.where(res == artist)[0]
            group_data = test_data[ind][result_indices]
            gen_scatters.append(ax.scatter(
                       group_data[..., 0],
                       group_data[..., 1],
                       group_data[..., 2],
                       facecolors=color,
                       lw=0.5,
                       alpha=0.98,
                       s=30
                       ))
    # ax_legend.legend(labels=leg)
    # ax_legend.legend(labels=leg, loc='upper right', bbox_to_anchor=(1, 1), ncol=1,
    #            bbox_transform=plt.gcf().transFigure)

    scatters = []
    for movement in movements.values():
        for artist, color in movement[0]:
            leg += [artist]
            result_indices = np.where(res == artist)[0]
            poss, nm = im_desc_per_artist[artist]
            for im_pos, n in zip(poss, nm):
                # im_pos = im_desc_per_artist[artist]
                scatters.append((ax.scatter(
                           im_pos[0],
                           im_pos[1],
                           im_pos[2],
                           # edgecolors='r',
                           facecolors=color,
                           lw=0.5,
                           alpha=0.9,
                           s=300),
                                 im_pos, n))

    def hover(event):
        if event.inaxes == ax:
            for sc_tup in scatters:
                cont, ind = sc_tup[0].contains(event)
                if cont:
                    choice = np.random.choice(ind['ind'])
                    im.set_data(image_dict[sc_tup[2]])

                    fig.canvas.draw_idle()
    def swap(event):
        if event.dblclick:
            for sc in gen_scatters:
                sc.set_alpha(np.abs(1-sc.get_alpha()))
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    connection_id = fig.canvas.mpl_connect('button_press_event', swap)
    # animation = FuncAnimation(fig, update, interval=1)
    # animation = ArtistAnimation(fig, image_dict.values(), interval=100, repeat=True)

    plt.show()

def demo():
    Centroid()

if __name__ == '__main__':


    DATA_FILE = "pca_models/pca_2src,num_of_artists=23,strategy1=hist_descauto_desc.npyfix.npy,strategy2=canny_descauto_desc.npyfix.npy"
    # load data
    data = np.load(DATA_FILE)
    targets = np.load(DATA_FILE + "targets.npy")
    src_paths = np.load(DATA_FILE + "src_paths.npy").astype(np.str)

    color_groups = movements
    all_artists = []
    for m in movements.values():
        for art, _ in m[0]:
            all_artists.append(art)


    demo()
    # training_scatter()

    # for i, im_path in enumerate(unique_im_paths):
    #     x, y, z = im_coords[i]
    #     img = io.imread(im_path)
    #     im = OffsetImage(img, zoom=0.02)
    #     ab = AnnotationBbox(im, (x, y),  frameon=False)
    #     ax.scatter(x, y, z)
    #     ax.add_artist(ab)


    # plt.show()






