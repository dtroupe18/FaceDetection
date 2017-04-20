"""
My idea is to use some principles from eigenfaces to better represent
the images that are faces. I will take all of the face images in the training set and
calculate the average face. I will then use this to train the perceptron
the testing images will then be compared to the average face. Hopefully, the
images with faces will have more in common then the images without faces
"""

import numpy as np


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
    # for x in average_face:
    #     print(x, ", ", end="")

    # subtract average face from every image in the training set
    centered_data = [x - average_face for x in training_data]
    np_array = np.array(centered_data)

    return np_array


def calculate_symmetry(images):
    """
    Faces should be horizontally symmetrical or at least close
    to horizontally symmetrical. For each row in the array representing the
    image I will calculate the number of points that are symmetrical. Since the
    image is 28x28 this will provide 28 separate numbers that measure the symmetry
    """

    # see how this does then consider looking at only the symmetry of the 1's (exclude all 0's)
    symmetry_scores = []
    for image in images:
        current_image_sym_scores = []
        for row in image:
            row_sym_score = 0
            first = False
            for i in range(30):
                if not first and row[i] == 1:
                    first = True
                    if row[i] == row[59 - i]:
                        row_sym_score += 1
                if first and row[i] != 0 and row[i] == row[59 - i]:
                    # print("row[", i, "] == row[", 59-i, "]")
                    row_sym_score += 1

            current_image_sym_scores.append(row_sym_score)
        # print(current_image_sym_scores)
        # print(len(current_image_sym_scores))
        symmetry_scores.append(current_image_sym_scores)

    return symmetry_scores
