from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D
import operator
import os


import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def load_data(strategy, library="artist_data", artist_filter=lambda artist:True, report=False):
    descriptor_list = []
    names = []
    for artist in filter(artist_filter, os.listdir(library)):
        desc_path = os.path.join(library, artist, "paint", "descriptor", strategy)
        try:
            descriptor_list.append(np.load(desc_path))
            names.append(artist)
        except:
            if report:
                print("Found no related descriptor {} for {} .".format(strategy, artist))
            continue
    return descriptor_list, names


def show_hist(rev):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ## the data
    N = len(rev)


    ## necessary variables
    ind = np.arange(N)  # the x locations for the groups
    width = 0.6  # the width of the bars

    ## the bars
    rects1 = ax.bar(ind, rev, width,
                    color='black',
                    # yerr=menStd,
                    error_kw=dict(elinewidth=2, ecolor='red'))
    ax.set_xlim(-width, len(ind) + width)
    # ax.set_ylim(0, 1)
    ax.set_ylabel('Count')
    ax.set_title('Hues Histogram')
    # xTickMarks = ['c ' + str(i) for i in range(256)]
    # ax.set_xticks(ind + width)
    # xtickNames = ax.set_xticklabels(xTickMarks)
    # plt.setp(xtickNames, rotation=45, fontsize=10)

    ## add a legend
    # ax.legend((rects1[0]), ('hues'))

    plt.show()

if __name__ == '__main__':
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    texture = "texture_patch_desc"
    canny = "canny_desc"
    canny2 = "canny_desc2"
    brief = "descriptor"
    rand = 'auto_desc'
    frequency = 'freq_desc'
    color = 'hist_desc'
    namespace = 'names.npy'
    strategy1 = color + rand + ".npy" + "fix.npy"

    strategy2 = canny + rand + ".npy" + "fix.npy"


    np_type = np.float64

    OPAC_ARTIST = [ "Personal Portfolio 08 - 10", "Personal Portfolio 12 - 15", "Personal Portfolio 16 - 18"]
    TESTING = True
    NUM_OF_COMPONENTS = 3
    NUM_OF_COMPONENTS_IGNORE = 0

    print("Loading data from file")
    desc1, names1 = load_data(strategy=strategy1, artist_filter=lambda artist: "Personal Portfolio" not in artist)
    desc2, names2 = load_data(strategy=strategy2, artist_filter=lambda artist: "Personal Portfolio" not in artist)
    portfolio_desc1, portfolio_names = load_data(strategy=strategy1, artist_filter=lambda artist: "Personal Portfolio" in artist)
    portfolio_desc2, _ = load_data(strategy=strategy2, artist_filter=lambda artist: "Personal Portfolio" in artist)

    src_paths_tup_artists, _ = load_data(namespace, artist_filter=lambda artist: "Personal Portfolio" not in artist)
    src_paths_tup_portfolio, _ = load_data(namespace, artist_filter=lambda artist: "Personal Portfolio" in artist)


    # for art in paths:
    #     print (art)
    #     # for desc in range(0, len(art), 10):
    #         # plt.imshow(art[desc])
    #         # plt.show()

    names = names1
    print("Preprocessing data")

    data1 = np.vstack(desc1).astype(np_type)
    data2 = np.vstack(desc2).astype(np_type)
    portfolio_data1 = np.vstack(portfolio_desc1).astype(np_type)
    portfolio_data2 = np.vstack(portfolio_desc2).astype(np_type)

    all_src_paths = np.concatenate(src_paths_tup_artists + src_paths_tup_portfolio).astype(np.str)

    colors_list = ['#1fd337', '#72b553', '#305b22', '#094c12', '#08aa1e', '#94bc82', '#47593f', '#513e0d', '#a58142',
                   '#ba7907', '#772975', '#d739db', '#c477c6', '#831fc4', '#ba74e8', '#855ba0', '#ede94e', '#d8cd79',
                   '#d8d313', '#dbd98a', '#f7e459', '#0b1054', '#409ce8']

    portfolio_colors = ["#a5100b", "#bf2e2a", "#fc0b05"]


    colors = {name: colors_list[i]
              for i, name in enumerate(filter(lambda name: not name.startswith("Personal Portfolio"),
                                              names))}
    for i, portfolio in enumerate(portfolio_names):
        colors[portfolio] = portfolio_colors[i]

    if color not in strategy1:

        data1 = data1.reshape((data1.shape[0], data1.shape[1] * data1.shape[2]))
        portfolio_data1 = portfolio_data1.reshape((portfolio_data1.shape[0],
                                                   portfolio_data1.shape[1] * portfolio_data1.shape[2]))

    if color not in strategy2:

        data2 = data2.reshape((data2.shape[0], data2.shape[1] * data2.shape[2]))
        portfolio_data2 = portfolio_data2.reshape((portfolio_data2.shape[0],
                                                   portfolio_data2.shape[1] * portfolio_data2.shape[2]))

    # n_data1 = normalize(data1)
    # n_port_data1 = normalize(portfolio_data1)
    ndata1 = normalize(np.vstack((data1, portfolio_data1)))
    training_data1 = ndata1[:data1.shape[0]]
    test_data1 = ndata1[data1.shape[0]:]

    # n_data2 = normalize(data2)
    # n_port_data2 = normalize(portfolio_data2)
    ndata2 = normalize(np.vstack((data2, portfolio_data2)))
    training_data2 = ndata2[:data1.shape[0]]
    test_data2 = ndata2[data1.shape[0]:]

    lens = [art.shape[0] for art in desc1] + [art.shape[0] for art in portfolio_desc1]

    targs = []
    mean_targs = []
    for i, name in enumerate(names + portfolio_names):
        targs += [name] * (lens[i])
        mean_targs += [name] * (lens[i] // 10)


    targets = pd.DataFrame(targs, columns=['target'])
    mean_targets = pd.DataFrame(mean_targs, columns=['target'])

    print("Running PCA")
    pca1 = PCA(n_components=2)

    principalComponents1 = pca1.fit_transform(training_data1)

    # for i in range(-10, 10):
    #
    #     rev = pca1.inverse_transform([i/10, i/10])
    #
    #     show_hist(rev)


    pca2 = PCA(n_components=1)

    principalComponents2 = pca2.fit_transform(training_data2)

    # for i in range(-10, 10):
    #     rev = pca2.inverse_transform([i / 100])
    #     plt.imshow(rev.reshape(128, 128), cmap='gray')
    #     plt.show()

    cropped_components = np.hstack((principalComponents1, principalComponents2))

    portfolio_points1 = pca1.transform(test_data1)
    portfolio_points2 = pca2.transform(test_data2)

    portfolio_points = np.hstack((portfolio_points1, portfolio_points2))

    print("Processing results of PCA")

    columns = ["principal component {}".format(i+1) for i in range(NUM_OF_COMPONENTS)]

    stack = np.vstack((cropped_components, portfolio_points))

    principalDfAll = pd.DataFrame(data=stack,
                               columns=columns)
    finalDfAll = pd.concat([principalDfAll, targets], axis=1)

    file_name = "pca_models/" + ",".join(("pca_2src",
                                          "num_of_artists={}".format(len(names)),
                                          "strategy1={}".format(strategy1),
                                          "strategy2={}".format(strategy2)))
    print("saving file: {}".format(file_name))
    np.save(file_name, stack)
    np.save(file_name + "targets", targs)
    np.save(file_name + "src_paths", all_src_paths)
    # finalDf.to_csv(file_name)

    data_means = np.mean(stack.reshape(stack.shape[0]//10, 10, NUM_OF_COMPONENTS), axis=1)

    principalDfPerIm = pd.DataFrame(data=data_means,
                               columns=columns)
    finalDfPerIm = pd.concat([principalDfPerIm, mean_targets], axis=1)



    print("Graphing")

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.set_xlabel('Hue Component 1', fontsize = 15)
    ax1.set_ylabel('Hue Component 2', fontsize = 15)
    ax1.set_zlabel('Canny Component', fontsize = 15)
    ax1.set_title('3 component PCA', fontsize=20)

    targets = names + portfolio_names
    for target, color in zip(targets, colors.values()):
        indicesToKeep = finalDfPerIm['target'] == target
        ax1.scatter(finalDfPerIm.loc[indicesToKeep, 'principal component 1']
                    , finalDfPerIm.loc[indicesToKeep, 'principal component 2']
                    , finalDfPerIm.loc[indicesToKeep, 'principal component 3']
                    , c=colors[target]
                    , s=50
                    , alpha=0.9 if not TESTING or target in OPAC_ARTIST else 0.7)
    ax1.set_alpha(1)
    ax1.legend(targets)
    ax1.grid()
    plt.show()

    fig = plt.figure(figsize = (8,6))
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.set_xlabel('Hue Component 1', fontsize = 15)
    ax2.set_ylabel('Hue Component 2', fontsize = 15)
    ax2.set_zlabel('Canny Component', fontsize = 15)
    ax2.set_title('3 component PCA - MEANS', fontsize = 20)


    for target, color in zip(targets, colors.values()):
        indicesToKeep = finalDfPerIm['target'] == target
        x = np.median(finalDfPerIm.loc[indicesToKeep, 'principal component 1'])
        y = np.median(finalDfPerIm.loc[indicesToKeep, 'principal component 2'])
        z = np.median(finalDfPerIm.loc[indicesToKeep, 'principal component 3'])
        ax2.scatter(x, y, z
                    , c=color
                    , s=200
                    , alpha=0.9 if (not TESTING or target in OPAC_ARTIST) else 0.4
                    )
        # x = np.mean(finalDf.loc[indicesToKeep, 'principal component 1'])
        # y = np.mean(finalDf.loc[indicesToKeep, 'principal component 2'])
        # z = np.mean(finalDf.loc[indicesToKeep, 'principal component 3'])
        # ax2.scatter(x, y, z
        #             , c=color
        #             , s=100
        #             , alpha=0.9 if (not TESTING or target in OPAC_ARTIST) else 0.4
        #             )
    for target, color in zip(targets, colors.values()):
        indicesToKeep = finalDfPerIm['target'] == target
        ax2.scatter(finalDfPerIm.loc[indicesToKeep, 'principal component 1']
                    , finalDfPerIm.loc[indicesToKeep, 'principal component 2']
                    , finalDfPerIm.loc[indicesToKeep, 'principal component 3']
                    , c=color
                    , s=50
                    , alpha=0.01)
    ax1.set_alpha(1)

    ax2.legend(targets)
    ax2.grid()
    plt.show()

    from sklearn.neighbors import NearestNeighbors
    # test_data_size = tst.shape[0]

    test_data = portfolio_points
    # test_data = tst

    X = cropped_components
    # X = data[:-test_data_size]
    # test_data = data[-test_data_size:]
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)
    # distances, indices = nbrs.kneighbors(X)
    test_distances, test_indices = nbrs.kneighbors(test_data)
    dic = {name: 0 for name in names}

    print ("Calculating success rate for test using Nearest Neighboirs")
    for name in finalDfPerIm.loc[test_indices.flatten()]['target']:
        dic[name] += 1
    m = test_indices.size
    for k, v in sorted(dic.items(), key=operator.itemgetter(0), reverse=True):
        print (k , "\t : \t", (int)((v / m) * 100) , "%")
    pass


    from sklearn.neighbors.nearest_centroid import NearestCentroid
    y = np.array(filter(lambda name: "Portfolio" in name, targs))
    clf = NearestCentroid()
    clf.fit(X, y)
    # NearestCentroid(metric='euclidean', shrink_threshold=None)
    print ("Calculating success rate for test using Centroids")

    res = clf.predict(test_data)
    dic = {r : 0 for r in res}
    for r in res:
        dic[r] += 1
    m = len(res)
    for k, v in sorted(dic.items(), key=operator.itemgetter(0), reverse=True):
        print (k , "\t : \t", (int)((v / m) * 100) , "%")
    pass



