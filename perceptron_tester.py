import load_face_data
from perceptron import Perceptron
import numpy as np
from sklearn.decomposition import PCA
import feature_extraction
from sklearn.preprocessing import StandardScaler


training_data, validation_data, test_data = load_face_data.load_data()

train_images = np.array(training_data[0])
train_labels = np.array(training_data[1])

test_images = np.array(test_data[0])
test_labels = np.array(test_data[1])

validation_images = np.array(validation_data[0])
validation_labels = np.array(validation_data[1])

# combine training and validation data to get better results
xyz = validation_data[0] + training_data[0]
abc = []
abc.extend(validation_data[1])
abc.extend(training_data[1])
train_and_valid_images = np.array(xyz)
train_and_valid_labels = np.array(abc)


def run_regular_perceptron(rate, epochs):
    perceptron = Perceptron(rate, epochs, "Perceptron 1")
    perceptron.fit(train_images, train_labels)
    results = perceptron.predict(test_images)
    perceptron.calculate_results(results, test_labels, 150)
    perceptron.graph_perceptron()


"""
A single perceptron performed very well on this binary classification achieving 100%
accuracy by the end of the 7th epoch on the training data.
For fun I reduced the problem down to a smaller number of dimensions (450) using PCA
from sklearn.
"""


def run_pca_perceptron(rate, epochs, dimensions):
    # Dimensionality Reduction down to 450 dimensions
    pca = PCA(n_components=dimensions)
    pca.fit(train_images)
    pca_train_images = pca.transform(train_images)
    pca_test_images = pca.transform(test_images)

    # try perceptron on reduced dimensionality data
    perceptron2 = Perceptron(rate, epochs, "Perceptron 2 PCA - ", rate)
    perceptron2.fit(pca_train_images, train_labels)
    # perceptron2.graph_perceptron()

    """
    Through testing I was able to determine that a reduction to 450 dimensions from the original 4200
    still allows the perceptron to converge. Thus it maintains 100% accuracy on the training data.
    """

    # Check Results on Test Data

    results_with_pca = perceptron2.predict(pca_test_images)
    perceptron2.calculate_results(results_with_pca, test_labels, 150)



"""
Interestingly enough the perceptron with the reduced dimensions actually out performed the
perceptron without dimensional reduction. 76 correct vs 77.

The problem here is that the perceptron is suffering from over fitting. That is the model is perfect
on the training data, but that does not linearly separate the test data. For this reason I think
more significant features need to be found to get a better model.

"""

def run_pca50_perceptron():
    # Try PCA with 50 features
    pca2 = PCA(n_components=50)
    pca2.fit(train_images)
    pca_train_images2 = pca2.transform(train_images)
    pca_test_images2 = pca2.transform(test_images)

    # try perceptron with 50 dimensions
    perceptron3 = Perceptron(0.01, 15, "Perceptron 3 PCA - 50")
    perceptron3.fit(pca_train_images2, train_labels)
    perceptron3.graph_perceptron()
    results_with_pca2 = perceptron3.predict(pca_test_images2)
    perceptron3.calculate_results(results_with_pca2, test_labels, 150)

"""
Again this didn't do too well because the model is still over fitting the training data. 51.33% the same as before.
"""


def run_train_validation_perceptron(rate, epochs):

    perceptron4 = Perceptron(rate, epochs, "Perceptron 4 Train & Validation")
    perceptron4.fit(train_and_valid_images, train_and_valid_labels)
    final_results = perceptron4.predict(test_images)
    perceptron4.calculate_results(final_results, test_labels, 150)
    perceptron4.graph_perceptron()



"""
Even with all of the validation images added into the training set the
perceptron is still only 57.77% accurate which isn't very good. I will have
to try and determine a way to extract features from the face images such that
they are more easily distinguished from random images.
"""


def run_sym_face_perceptron():
    training_data_2d, validation_data_2d, test_data_2d = load_face_data.load_data(two_d=True)

    train_images_2d = np.array(training_data_2d[0])
    train_labels_2d = np.array(training_data_2d[1])
    standardized_images = StandardScaler().fit_transform(train_images_2d)


    test_images_2d = np.array(test_data_2d[0])
    test_labels_2d = np.array(test_data_2d[1])
    standardized_test_images = StandardScaler().fit_transform(test_images_2d)

    validation_images_2d = np.array(validation_data_2d[0])
    validation_labels_2d = np.array(validation_data_2d[1])

    sym_perceptron = Perceptron(3, 100, "Sym Face")
    sym_perceptron.fit(standardized_images, train_labels)
    sym_results = sym_perceptron.predict(standardized_test_images)
    sym_perceptron.calculate_results(sym_results, test_labels, 150)
    sym_perceptron.graph_perceptron()

    """
    Symmetric face feature extraction didn't work any better than the standard perceptron. Additionally,
    it takes significantly longer to run. The main issue here is that the perceptron is unable
    to converge. Thus the feature extraction is not distinguishing the face and not-face very well.

    Standardizing the images made the perceptron run faster, but it still doesn't converge
    """


def run_average_face_perceptron(rate, epochs, standardize=False):

    # extract just the face images from both the training and validation datasets
    faces = feature_extraction.get_face_images(train_and_valid_images, train_and_valid_labels)

    # find the average of all the face images and subtract is from every image
    # this also centers all of the images
    new_training_images = feature_extraction.find_average_face(faces, train_images)

    # Find the average image in the testing set and subtract it from every image
    # this allows us to compare the testing images more accurately
    new_test_images = feature_extraction.find_average_face(test_images, test_labels)

    if standardize:
        standardized_train_images = StandardScaler().fit_transform(new_training_images)
        standardized_test_images = StandardScaler().fit_transform(new_test_images)
        average_face_perceptron = Perceptron(rate, epochs, "Perceptron Average Face (Standardized)")
        average_face_perceptron.fit(standardized_train_images, train_labels)
        average_results = average_face_perceptron.predict(standardized_test_images)
        average_face_perceptron.calculate_results(average_results, test_labels, 150)
        average_face_perceptron.graph_perceptron()

    else:
        average_face_perceptron = Perceptron(rate, epochs, "Perceptron Average Face")
        average_face_perceptron.fit(new_training_images, train_labels)
        average_results = average_face_perceptron.predict(new_test_images)
        average_face_perceptron.calculate_results(average_results, test_labels, 150)
        average_face_perceptron.graph_perceptron()


"""
Average faces gets 100% accuracy in both cases. I am using both the training and validation images
to train the perceptron. Standardizing the values just makes the perceptron converge faster nine
epochs versus eighteen.
"""


# run_regular_perceptron(0.01, 20)
# run_train_validation_perceptron(0.01, 25)
# run_sym_face_perceptron()
run_average_face_perceptron(0.01, 20)
run_average_face_perceptron(0.01, 12, standardize=True)

