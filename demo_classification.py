import os
import numpy as np
from scipy.ndimage import imread
from scipy.spatial.distance import cdist


# Parameters
nrun = 20  # Number of classification runs
path_to_script_dir = os.path.dirname(os.path.realpath(__file__))
path_to_all_runs = os.path.join(path_to_script_dir, 'all_runs')
fname_label = 'class_labels.txt'  # Where class labels are stored for each run


def classification_run(folder, f_load, f_cost, ftype='cost'):
    # Compute error rate for one run of one-shot classification
    #
    # Input
    #  folder : contains images for a run of one-shot classification
    #  f_load : itemA = f_load('file.png') should read in the image file and
    #           process it
    #  f_cost : f_cost(itemA,itemB) should compute similarity between two
    #           images, using output of f_load
    #  ftype  : 'cost' if small values from f_cost mean more similar,
    #           or 'score' if large values are more similar
    #
    # Output
    #  perror : percent errors (0 to 100% error)
    #
    assert ftype in {'cost', 'score'}

    with open(os.path.join(path_to_all_runs, folder, fname_label)) as f:
        pairs = [line.split() for line in f.readlines()]
    # Unzip the pairs into two sets of tuples
    test_files, train_files = zip(*pairs)

    answers_files = list(train_files)  # Copy the training file list
    test_files = sorted(test_files)
    train_files = sorted(train_files)
    n_train = len(train_files)
    n_test = len(test_files)

    # Load the images (and, if needed, extract features)
    train_items = [f_load(os.path.join(path_to_all_runs, f))
                   for f in train_files]
    test_items = [f_load(os.path.join(path_to_all_runs, f))
                  for f in test_files]

    # Compute cost matrix
    costM = np.zeros((n_test, n_train))
    for i, test_i in enumerate(test_items):
        for j, train_j in enumerate(train_items):
            costM[i, j] = f_cost(test_i, train_j)
    if ftype == 'cost':
        y_hats = np.argmin(costM, axis=1)
    elif ftype == 'score':
        y_hats = np.argmax(costM, axis=1)
    else:
        # This should never be reached due to the assert above
        raise ValueError('Unexpected ftype: {}'.format(ftype))

    # compute the error rate by counting the number of correct predictions
    correct = len([1 for y_hat, answer in zip(y_hats, answers_files)
                   if train_files[y_hat] == answer])
    pcorrect = correct / float(n_test)  # Python 2.x ensure float division
    perror = 1.0 - pcorrect
    return perror * 100


def modified_hausdorf_distance(itemA, itemB):
    # Modified Hausdorff Distance
    #
    # Input
    #  itemA : [n x 2] coordinates of black pixels
    #  itemB : [m x 2] coordinates of black pixels
    #
    #  M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
    #  International Conference on Pattern Recognition, pp. 566-568.
    #
    D = cdist(itemA, itemB)
    mindist_A = D.min(axis=1)
    mindist_B = D.min(axis=0)
    mean_A = np.mean(mindist_A)
    mean_B = np.mean(mindist_B)
    return max(mean_A, mean_B)


def load_img_as_points(filename):
    # Load image file and return coordinates of black pixels in the binary image
    #
    # Input
    #  filename : string, absolute path to image
    #
    # Output:
    #  D : [n x 2] rows are coordinates
    #
    I = imread(filename, flatten=True)
    # Convert to boolean array and invert the pixel values
    I = ~np.array(I, dtype=np.bool)
    # Create a new array of all the non-zero element coordinates
    D = np.array(I.nonzero()).T
    return D - D.mean(axis=0)


# Main function
if __name__ == "__main__":
    #
    # Running this demo should lead to a result of 38.8% average error rate.
    #
    #   M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
    #     International Conference on Pattern Recognition, pp. 566-568.
    #
    # ** Models should be trained on images in 'images_background' directory to
    #    avoid using images and alphabets used in the one-shot evaluation **
    #
    print('One-shot classification demo with Modified Hausdorff Distance')
    perror = np.zeros(nrun)
    for r in range(nrun):
        perror[r] = classification_run('run{:02d}'.format(r + 1),
                                       load_img_as_points,
                                       modified_hausdorf_distance,
                                       'cost')
        print(' run {:02d} (error {:.1f}%)'.format(r, perror[r]))
    total = np.mean(perror)
    print('Average error {:.1f}%'.format(total))
