from Visuals import ART_MOVEMENTS, centroidVisualization, trainingVisualization
import numpy as np

DATA_FILE   = "demo_data"
DATA_EXT    = ".npy"
LEGEND_PATH = "pie_legend.png"

def demo(data, targets, src_paths, legend_path ):
    trainingVisualization(data, targets)
    centroidVisualization(data, targets, src_paths, legend_path)


if __name__ == '__main__':

    # load data
    data = np.load(DATA_FILE + DATA_EXT)
    targets = np.load(DATA_FILE + "," + "targets" + DATA_EXT)
    src_paths = np.load(DATA_FILE + "," + "src_paths" + DATA_EXT).astype(np.str)

    color_groups = ART_MOVEMENTS
    all_artists = []
    for m in ART_MOVEMENTS.values():
        for art, _ in m[0]:
            all_artists.append(art)

    # run demo
    demo(data, targets, src_paths, LEGEND_PATH)
