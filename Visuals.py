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
from sklearn.neighbors import NearestNeighbors

from matplotlib.offsetbox import *

ART_MOVEMENTS = {
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

color_groups = ART_MOVEMENTS
all_artists = []
for m in ART_MOVEMENTS.values():
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

def trainingVisualization(data, targets, with_portfolio=False):
    """
    Visual the data of all the training artists used to create the original mapping.
    """
    ALPHA_LOW = 0.02
    ALPHA_SEMI = 0.3
    ALPHA_HIGH = 1-ALPHA_LOW

    TITLE_FONT  = { 'fontsize': 25,
                    'fontweight': 1.5,
                    'verticalalignment': 'baseline',
                    'horizontalalignment': 'left'}

    fig = plt.figure(figsize=(10, 6), tight_layout=False)
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.autoscale()
    medians = dict()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel('Hue Component 1', fontsize=15)
    ax.set_ylabel('Hue Component 2', fontsize=15)
    ax.set_zlabel('Canny Component', fontsize=15)
    # ax.autoscale()
    if with_portfolio:
        ax.set_title("Training Data \n - Personal Portfolio Superimposed ", loc='left', fontdict=TITLE_FONT)
    else:
        ax.set_title("Training Data", loc='right', fontdict=TITLE_FONT)

    leg = []
    for group in color_groups:
        if not with_portfolio and "Portfolio" in group:
            continue
        for artist, color in ART_MOVEMENTS[group][0]:
            leg.append(artist)
            indices = np.where(targets == artist)[0]
            group_data = data[indices]
            x = np.median(group_data[..., 0])
            y = np.median(group_data[..., 1])
            z = np.median(group_data[..., 2])
            medians[artist] = np.array((x, y, z))
            ax.scatter(x, y, z,
                       edgecolors=ART_MOVEMENTS[group][1],
                       facecolors=color,
                       lw=0.5,
                       alpha=0.9,
                       s=200
                       )
    ax.legend(labels=leg, loc='upper right', bbox_to_anchor=(1, 1), ncol=1,
              bbox_transform=plt.gcf().transFigure)

    gen_scatters = []
    for group in color_groups:
        if not with_portfolio and "Portfolio" in group:
            continue
        a = ALPHA_LOW if "Portfolio" not in group else ALPHA_SEMI
        for artist, color in ART_MOVEMENTS[group][0]:
            leg.append(artist)
            indices = np.where(targets == artist)[0]
            group_data = data[indices]
            gen_scatters.append((ax.scatter(group_data[..., 0],
                                           group_data[..., 1],
                                           group_data[..., 2],
                                           edgecolors=ART_MOVEMENTS[group][1],
                                           facecolors=color,
                                           lw=0.5,
                                           alpha=a,
                                           s=20
                                           )
                                , artist))

    def swap(event):
        if event.dblclick:
            high = False
            for sc, artist in gen_scatters:
                sc.set_alpha(np.abs(1 - sc.get_alpha()))
                high = sc.get_alpha() > ALPHA_LOW
            if high:
                ax.set_title("All Artists", loc='left', fontdict=TITLE_FONT)
            else:
                ax.set_title("Medians Per Artist", loc='left', fontdict=TITLE_FONT)

        elif event.inaxes == ax:
            for sc, artist in np.random.permutation(gen_scatters):
                cont, ind = sc.contains(event)
                if cont:
                    choice = np.random.choice(ind['ind'])
                    sc.set_alpha(ALPHA_HIGH)
                    for other_sc, _ in filter(lambda s: s[0] != sc, gen_scatters):
                        other_sc.set_alpha(ALPHA_LOW)
                    ax.set_title(artist, loc='left', fontdict=TITLE_FONT)
                    break
            else:
                pass
                for sc, artist in gen_scatters:
                    sc.set_alpha(ALPHA_LOW)
                ax.set_title("Medians Per Artist", loc='left', fontdict=TITLE_FONT)

        fig.canvas.draw_idle()

    connection_id = fig.canvas.mpl_connect('button_press_event', swap)

    plt.show()
    plt.close(fig)


def centroidVisualization(data, targets, src_paths, legend_path="presentation/pie legend.png"):
    """
    Visualize the new data (Personal Portfolio) on the mapping of all the artists, by coloring the new data points with
    the artist's colors. The centroid decides for each data point which artist is fits to, and then a tally is performed
    on a per-image basis.
    """

    global my_indices, training_data, labels, test_data, im_coords, im_paths, ind, s_im_paths, s_im_coords, unique_im_paths, unique_indices

    # Create new Figure and an Axes which fills it.
    fig = plt.figure(figsize=(12, 6), tight_layout=False)
    # ax = p3.Axes3D(fig)
    ax = fig.add_subplot(122, projection='3d')
    ax_im = fig.add_subplot(221)
    ax_legend = fig.add_subplot(223)
    ax_legend.imshow(plt.imread(legend_path))

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
    ax.set_title('Portfolio Analysis Using Centroid Clustering', fontsize=20)

    # my_indices = np.where(is_portfolio(targets))[0]

    clf = NearestCentroid()
    # training_data = data[np.where(np.bitwise_not(is_portfolio(targets)))[0]]
    # labels = targets[np.where(np.bitwise_not(is_portfolio(targets)))[0]]

    # test_data = data[my_indices]

    clf.fit(training_data, labels)

    # im_paths = src_paths[my_indices]
    # im_coords = data[my_indices]
    # ind = np.argsort(im_paths)
    # s_im_paths = im_paths[ind]
    # s_im_coords = im_coords[ind]

    # unique_im_paths, unique_indices = np.unique(list(map(parse_path, s_im_paths)), return_index=True)
    dest_coords = s_im_coords[unique_indices]

    print("Calculating Centroids...")

    res = clf.predict(test_data[ind])

    im_desc_per_artist = {artist : ([], []) for artist in all_artists}
    for i, index in enumerate(unique_indices[:-1]):
        r = res[index : unique_indices[i+1]]
        median = np.median(s_im_coords[index: unique_indices[i+1]], axis=0)
        n, c = np.unique(r, return_counts=True)
        t = np.argmax(c)
        im_desc_per_artist[n[t]][0].append(median)
        im_desc_per_artist[n[t]][1].append(unique_im_paths[i])

    print("Graphing test data...")

    image_dict = {u: plt.imread(u) for u in unique_im_paths}
    im = ax_im.imshow(image_dict[unique_im_paths[0]])

    leg = []
    gen_scatters = []

    for movement in ART_MOVEMENTS.values():
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
    for movement in ART_MOVEMENTS.values():
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
    plt.close(fig)

def nearestNeighborsVisualization(data, targets, src_paths, legend_path="presentation/pie legend.png"):
    """
    Visualize the new data (Personal Portfolio) on the mapping of all the artists, by coloring the new data points with
    the artist's colors. The centroid decides for each data point which artist is fits to, and then a tally is performed
    on a per-image basis.
    """
    global my_indices, training_data, labels, test_data, im_coords, im_paths, ind, s_im_paths, s_im_coords, unique_im_paths, unique_indices

    # Create new Figure and an Axes which fills it.
    fig = plt.figure(figsize=(12, 6), tight_layout=False)
    ax = fig.add_subplot(122, projection='3d')
    ax_im = fig.add_subplot(221)
    ax_legend = fig.add_subplot(223)
    ax_legend.imshow(plt.imread(legend_path))

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
    ax.set_title('Portfolio Analysis Using Nearest Neighbors', fontsize=20)

    knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')

    nbrs = knn.fit(training_data, labels)


    # unique_im_paths, unique_indices = np.unique(list(map(parse_path, s_im_paths)), return_index=True)
    dest_coords = s_im_coords[unique_indices]

    print("Calculating Nearest Neighbors...")

    # res = knn.predict(test_data[ind])
    res_distances, res_indices = nbrs.kneighbors(test_data[ind])

    res_by_name = targets[res_indices]

    im_desc_per_artist = {artist : ([], []) for artist in all_artists}


    for i, index in enumerate(unique_indices[:-1]):
        relevent_for_im = res_by_name[index : unique_indices[i+1]]
        median = np.median(s_im_coords[index: unique_indices[i+1]], axis=0)
        candidates = np.unique(relevent_for_im)
        weight_by_distance = [np.sum(res_distances[np.where(res_by_name == artist)]) for artist in candidates]
        t = np.argmin(weight_by_distance)
        winner = candidates[t]
        im_desc_per_artist[winner][0].append(median)
        im_desc_per_artist[winner][1].append(unique_im_paths[i])

    print("Graphing test data...")

    image_dict = {u: plt.imread(u) for u in unique_im_paths}
    im = ax_im.imshow(image_dict[unique_im_paths[0]])

    leg = []
    gen_scatters = []

    for movement in ART_MOVEMENTS.values():
        for artist, color in movement[0]:
            leg += [artist]
            # result_indices = np.where(res_by_name == artist)[np.argmin(res_distances, axis=1)]
            result_indices = np.where(res_by_name == artist)[0]

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
    for movement in ART_MOVEMENTS.values():
        for artist, color in movement[0]:
            leg += [artist]
            # result_indices = np.where(res == artist)[0]
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
    plt.close(fig)


def setup(data, targets, src_paths):
    global my_indices, training_data, labels, test_data, im_coords, im_paths, ind, s_im_paths, s_im_coords, unique_im_paths, unique_indices

    print("Loading and setting up data...")
    my_indices = np.where(is_portfolio(targets))[0]
    training_data = data[np.where(np.bitwise_not(is_portfolio(targets)))[0]]
    labels = targets[np.where(np.bitwise_not(is_portfolio(targets)))[0]]

    test_data = data[my_indices]

    im_paths = src_paths[my_indices]
    im_coords = data[my_indices]
    ind = np.argsort(im_paths)
    s_im_paths = im_paths[ind]
    s_im_coords = im_coords[ind]

    unique_im_paths, unique_indices = np.unique(list(map(parse_path, s_im_paths)), return_index=True)
    print("Notice: Double click to add/reduce noise")


if __name__ == '__main__':
    DATA_FILE = "pca_models/pca_2src,num_of_artists=23,strategy1=hist_descauto_desc.npyfix.npy,strategy2=canny_descauto_desc.npyfix.npy"
    # load data
    data = np.load(DATA_FILE)
    targets = np.load(DATA_FILE + "targets.npy")
    src_paths = np.load(DATA_FILE + "src_paths.npy").astype(np.str)

    color_groups = ART_MOVEMENTS
    all_artists = []
    for m in ART_MOVEMENTS.values():
        for art, _ in m[0]:
            all_artists.append(art)

    trainingVisualization(data, targets)
    centroidVisualization(data, targets, src_paths)







