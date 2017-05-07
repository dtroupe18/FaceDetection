"""
Training data needs to be transformed into
[1, 0, 0, 1, 0, label]
"""

import math
import load_face_data
import numpy as np

def separate_by_label(dataset):
    """

    :param dataset: two dimensional list of data values
    :return: dictionary where labels are keys and
    values are the data points with that label
    """
    separated = {}
    for x in range(len(dataset)):
        row = dataset[x]
        if row[-1] not in separated:
            separated[row[-1]] = []
        separated[row[-1]].append(row)

    return separated


def calc_mean(lst):
    return np.mean(lst, 0).tolist()
    # return float(sum(lst) / float(len(lst)))


def calc_standard_deviation(lst):
    avg = calc_mean(lst)
    variance = sum([pow(x - avg, 2) for x in lst]) / float(len(lst) - 1)

    return math.sqrt(variance)


def summarize_data(lst):
    """
    Calculate the mean and standard deviation for each attribute
    :param lst: list
    :return: list with mean and standard deviation for each attribute
    """

    summaries = [(calc_mean(attribute), calc_standard_deviation(attribute)) for attribute in zip(*lst)]
    del summaries[-1]
    return summaries


def summarize_by_label(data):
    """
    Method to summarize the attributes for each label

    :param data:
    :return: dict label: [(atr mean, atr stdv), (atr mean, atr stdv)....]
    """
    separated_data = separate_by_label(data)
    summaries = {}
    for label, instances in separated_data.items():
        summaries[label] = summarize_data(instances)
    return summaries


def calc_probability(x, mean, standard_deviation):
    """

    :param x: value
    :param mean: average
    :param standard_deviation: standard deviation
    :return: probability of that value given a normal distribution
    """
    # e ^ -(y - mean)^2 / (2 * (standard deviation)^2)
    # print("STANDARD DEVIATION: ", standard_deviation)
    if standard_deviation == 0:
        s_d = 0.01
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(s_d, 2))))
        # ( 1 / sqrt(2π) ^ exponent
        return (1 / (math.sqrt(2 * math.pi) * s_d)) * exponent

    else:
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(standard_deviation, 2))))
        # ( 1 / sqrt(2π) ^ exponent
        return (1 / (math.sqrt(2 * math.pi) * standard_deviation)) * exponent


def calc_label_probabilities(summaries, input_vector):
    """
    the probability of a given data instance is calculated by multiplying together
    the attribute probabilities for each class. The result is a map of class values
    to probabilities.

    :param summaries:
    :param input_vector:
    :return: dict
    """
    probabilities = {}
    for label, label_summaries in summaries.items():
        probabilities[label] = 1
        for i in range(len(label_summaries)):
            mean, standard_dev = label_summaries[i]
            x = input_vector[i]
            probabilities[label] *= calc_probability(x, mean, standard_dev)

    return probabilities


def predict(summaries, input_vector):
    """
    Calculate the probability of a data instance belonging
    to each label. We look for the largest probability and return
    the associated class.
    :param summaries:
    :param input_vector:
    :return:
    """
    probabilities = calc_label_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for label, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = label

    return best_label


def get_predictions(summaries, test_set):
    """
     Make predictions for each data instance in our
     test dataset
    """

    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)

    return predictions


def get_accuracy(test_set, predictions):
    """
    Compare predictions to class labels in the test dataset
    and get our classification accuracy
    """
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1

    return (correct / float(len(test_set))) * 100


def main(feature_extraction=False):
    # prepare data
    if feature_extraction:
        training_set, test_set = load_face_data.format_dataset_naive_bayes(True)
    else:
        training_set, test_set = load_face_data.format_dataset_naive_bayes()

    print("Size of Training Set: ", len(training_set))
    print("Size of Testing Set: ", len(test_set))

    # create model
    summaries = summarize_by_label(training_set)

    # test mode
    predictions = get_predictions(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}%'.format(accuracy))


main()

"""
This gets poor results around 51.33% accuracy because many features have
a standard deviation of zero and I am not sure how to adjust the model
in order to get better results. This should be changed from using a
normal distribution to using conditional probability. This eliminates
the need to calculate the standard deviation.
"""