"""
Each face is 60x70 and from the Berkley classification project
which took the photos from the Cal-tech 101 data set

There are 451 training images
150 test images && 301 Validation images
again all images are stored as a text file such as

                                       ###
                          ###   ###
                        ##         ####
                      ##               ###
                     #                    ###
                    #                        #
                    #                         #     ###
                #   #                          #   #   ####
               #   #                            ###
              #    #          ##                 #
              #    #         #  #                 #
             #     #    #    #   ###               #
             #     #   #     #  #   #              #
            #      # ##      #  #   #              #
            #       #       #        #              ###
           #                #   #    #                 #
          #                 #  #     #                  #
          #                 #  #     #                   #
         #                  #  #     #                    #
         #            #    #    #     #                   #
         #            #    #    #     #                  #
        #            #     #     #    #                 #
        #           #      #     #     #               #
        #          #       #     #      #             #
        #        ##       #      #       ##           #
       #        #         #     #          #           #
       #        #        #      #  #        #          #
       #        #       #      #   #        #          #
       #        #      #       #   #        #          #
       #        #  ####        #    #####   #           #
       #        #              #           #            #
       #        #         #    #  #        #            #
       #        #         #    #  #        #            #
        #       #         #    #  #        #            #
        #       #         #    #  #       #             #
        #        #        #    #  #       #              #
        #      #  #      #     #  #       #              #
        #         #     ##     #   #      #             #
         #      #    ###  #    #    ###   #             #
         #       #      #      #    #      #            #
          ###    #      #      #   #       #  #         #
             #   #       #       ##        #  #         #
             #   #       ###    #   #     #    #        #
          #   #   #     #   ####    #     #    #       #
          #   #   #     #   #   #  #     #     #      #
          #   #   #      ## #   # #      #      #     #
          #   #    #       #     #      #      # ##  #
          #  ##    #                    #     #      #
          # #  #   #                   #      #      #
          #    #   #   #              #      #      #
          #        #    #          ###             #
          #        #     #        #   #           #
           #        #     #      #    #           #
            #       #      ### ##     #         ##
             #      #         #       #     ####
              ####  #          #   #  #              ######
                    #      ########   #     #     ###
                    #                 #      #####
                    #                  #     #
                    #                   ###
                    #                      #
                   #                        ##    ##
    ####           #                          #  #  ##
  ##    #          #                           ##     ####
 #       #                                    #           #
          #####                              #
               #############           ######

    This file will load all of the images into an array
    where each image is inside a one dimensional list
    with 4,200 (60x70) entries

"""

import itertools
import os
import numpy as np
import feature_extraction
from sklearn.preprocessing import StandardScaler


def read_lines(filename):
    if os.path.exists(filename):
        return [line[:-1] for line in open(filename).readlines()]
    else:
        return "Path not found"


def create_images(data, height, width, n):
    """
    Takes the data and returns a list of lists where each entry
    is one image stored in an array of length 4,200 the pixels are
    marked as on or off. That is blank spaces get a 0 and pixels
    with part of the image are 1.
    """

    all_images = []
    for i in range(n):
        current_int_image = []
        for j in range(height):
            current_line = data.pop()
            current_integer_line = []
            for k in range(len(current_line)):
                if current_line[k] == ' ':
                    current_integer_line.append(0)
                elif current_line[k] == '#':
                    current_integer_line.append(1)

            current_int_image.append(current_integer_line)
        if len(current_int_image[-1]) < width:
            break
        """ No Feature Extraction """
        # store the 0, 1's for the image in a one dimensional array
        flat = list(itertools.chain(*current_int_image))
        all_images.append(flat)

    # convert to numpy ndarray
    # np_array = np.array(all_images)

    return all_images


def create_2d_images(data, height, width, n):
    """
    Takes the data and returns a list of lists where each entry
    is one image stored in a 2 dimensional array 60x70 the pixels are
    marked as on or off. That is blank spaces get a 0 and pixels
    with part of the image are 1.
    """
    all_images = []
    for i in range(n):
        current_int_image = []
        for j in range(height):
            current_line = data.pop()
            current_integer_line = []
            for k in range(len(current_line)):
                if current_line[k] == ' ':
                    current_integer_line.append(0)
                elif current_line[k] == '#':
                    current_integer_line.append(1)

            current_int_image.append(current_integer_line)
        if len(current_int_image[-1]) < width:
            break

        all_images.append(current_int_image)
        """ Feature Extraction """
        symmetry_scores = feature_extraction.calculate_symmetry(all_images)

    return symmetry_scores


def load_labels(filename):
    """
    Labels for the numbers are obvious 0 - 1
    1 for a face
    0 for not a face
    """
    data = read_lines(filename)
    labels = []
    for line in data:
        if line == '':
            break
        labels.append(int(line))

    np_array = np.array(labels)

    return np_array


def load_data():
    """Return the number image data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    array with 451 entries.  Each entry is, in turn, an array with
    4,200 values, representing the 60 * 70 = 4,200 pixels in a
    single image. On pixels are a 1 and off pixels are a 0.

    The second entry in the ``training_data`` tuple is an array
    containing 451 entries.  Those entries are just the digit
    values or labels 0 - 1 for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 1,000 images.
    """

    train_data = read_lines("facedata/facedatatrain")
    train_images = create_images(train_data, 70, 60, 451)
    train_labels = load_labels("facedata/facedatatrainlabels")

    val_data = read_lines("facedata/facedatavalidation")
    val_images = create_images(val_data, 70, 60, 301)
    val_labels = load_labels("facedata/facedatavalidationlabels")

    test_data = read_lines("facedata/facedatatest")
    test_images = create_images(test_data, 70, 60, 150)
    test_labels = load_labels("facedata/facedatatestlabels")

    training_data = (train_images, train_labels)
    validation_data = (val_images, val_labels)
    testing_data = (test_images, test_labels)

    return training_data, validation_data, testing_data


def get_face_images(images, labels):
    # face has a value of 1
    index_of_faces = []
    face_images = []

    for i in range(len(labels)):
        if labels[i] == 1:
            index_of_faces.append(i)

    for index in index_of_faces:
        face_images.append(images[index])

    # 217 / 451 faces in training data
    return face_images


def find_average_face(face_images, training_data):
    # find the mean face and subtract it from every image in the dataset

    # compute the average face
    average_face = np.mean(face_images, 0)

    # subtract average face from every image in the training set
    centered_data = training_data - average_face

    return centered_data


def format_dataset_knn(features=False):
    # load face images
    training_data, validation_data, test_data = load_data()
    test_labels = test_data[1]
    train_labels = training_data[1]

    if features:
        faces = feature_extraction.get_face_images(training_data[0], training_data[1])
        new_train_images = feature_extraction.find_average_face(faces, training_data[0])
        new_test_images = feature_extraction.find_average_face(test_data[0], test_data[1])
        train_images = StandardScaler().fit_transform(new_train_images).tolist()
        test_images = StandardScaler().fit_transform(new_test_images).tolist()

    else:
        train_images = StandardScaler().fit_transform(training_data[0]).tolist()
        test_images = StandardScaler().fit_transform(test_data[0]).tolist()

    # add label to image data so the last element is "Face" or "Not-Face"
    for x in range(len(train_labels)):
        if train_labels[x] == 1:
            train_images[x].append('Face')
        else:
            train_images[x].append('Not-Face')

    for x in range(len(test_labels)):
        if test_labels[x] == 1:
            test_images[x].append('Face')
        else:
            test_images[x].append('Not-Face')
            # print(train_images[0])

    return train_images, test_images


def format_dataset_naive_bayes(features=False):
    # load face images
    training_data, validation_data, test_data = load_data()
    test_labels = test_data[1]
    train_labels = training_data[1]

    if features:
        print("Data Loading...")
        faces = feature_extraction.get_face_images(training_data[0], training_data[1])
        new_train_images = feature_extraction.find_average_face(faces, training_data[0]).tolist()
        new_test_images = feature_extraction.find_average_face(test_data[0], test_data[1]).tolist()
        train_images = StandardScaler().fit_transform(new_train_images).tolist()
        test_images = StandardScaler().fit_transform(new_test_images).tolist()

    else:
        train_images = StandardScaler().fit_transform(training_data[0]).tolist()
        test_images = StandardScaler().fit_transform(test_data[0]).tolist()

    # add label to image data so the last element is "Face" or "Not-Face"
    for x in range(len(train_labels)):
        if train_labels[x] == 1:
            train_images[x].append(1)
        else:
            train_images[x].append(0)

    for x in range(len(test_labels)):
        if test_labels[x] == 1:
            test_images[x].append(1)
        else:
            test_images[x].append(0)
            # print(train_images[0])

    return train_images, test_images
