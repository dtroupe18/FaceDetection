"""
Simple KNN using euclidean distance or cosine distance
"""

import math
import operator
from scipy.spatial.distance import cosine
import load_face_data
import feature_extraction
from sklearn.preprocessing import StandardScaler
import numpy as np


def format_dataset(features=False):
    # load face images
    training_data, validation_data, test_data = load_face_data.load_data()
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
            train_images[x].append("Face")
        else:
            train_images[x].append("Not-Face")

    for x in range(len(test_labels)):
        if test_labels[x] == 1:
            test_images[x].append("Face")
        else:
            test_images[x].append("Not-Face")

    # print(train_images[0])

    return train_images, test_images


def euclidean_distance(instance1, instance2, length):
    """
        Calculates the euclidean distance between two points
    :param instance1: data point 1
    :param instance2: data point 2
    :param length: how many dimensions to evaluate
    :return: distance
    """
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def cosine_distance(instance1, instance2):
    # Assuming the last element of each instance is the label or class
    p1 = instance1[:-1]
    p2 = instance2[:-1]
    return cosine(p1, p2)


def get_neighbors(training_set, test_instance, k, cosine_d=False):
    """
    Give a test instance return the k nearest neighbors in
    the training set

    :param training_set: list of training data points
    :param test_instance: single point in the test set
    :param k: number of neighbors to use
    :return: list of k closest data points

    """
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        if cosine_d:
            dist = cosine_distance(test_instance, training_set[x])
            distances.append((training_set[x], dist))
        else:
            dist = euclidean_distance(test_instance, training_set[x], length)
            distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    # print(neighbors)
    return neighbors


def get_response(neighbors):
    """
    The last attribute of each neighbor is the label
    Ex: [1, 1, 1, a]
    :param neighbors: list of closest points to a data point
    :return:
    """
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def get_accuracy(test_set, predictions):
    """

    :param test_set: testing dataset
    :param predictions: labels based on kNN
    :return: percent correct
    """
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def main(k, feature_extraction=False, cosine_d=False):
    # prepare data
    if feature_extraction:
        training_set, test_set = format_dataset(True)
    else:
        training_set, test_set = format_dataset()

    print('Train set: ' + repr(len(training_set)))
    print('Test set: ' + repr(len(test_set)))
    # generate predictions
    predictions = []

    if cosine_d:
        for x in range(len(test_set)):
            neighbors = get_neighbors(training_set, test_set[x], k)
            result = get_response(neighbors)
            predictions.append(result)
            print('> predicted=' + repr(result) + ', actual=' + repr(test_set[x][-1]))
        accuracy = get_accuracy(test_set, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')

    else:
        for x in range(len(test_set)):
            neighbors = get_neighbors(training_set, test_set[x], k)
            result = get_response(neighbors)
            predictions.append(result)
            print('> predicted=' + repr(result) + ', actual=' + repr(test_set[x][-1]))
        accuracy = get_accuracy(test_set, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')


main(2, feature_extraction=True, cosine_d=True)


"""
It only seems to make sense to use 2 clusters since we only have two possible labels. When I tested clusters larger
than 2 the accuracy was reduced.

Statistics using 100% of the training data

k = 2, Average-Face, Cosine = 100%
k = 2, Average-Face, Euclidean = 100%
k = 2, No Features, Cosine = 50.667%
k = 2, No Features, Euclidean = 51.333%
"""
