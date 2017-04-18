import load_face_data
from perceptron import Perceptron
import numpy as np
from sklearn.decomposition import PCA

training_data, validation_data, test_data = load_face_data.load_data()

train_images = np.array(training_data[0])
train_labels = np.array(training_data[1])

test_images = np.array(test_data[0])
test_labels = np.array(test_data[1])

validation_images = np.array(validation_data[0])
validation_labels = np.array(validation_data[1])

perceptron = Perceptron(0.01, 9, "Perceptron 1")
perceptron.fit(train_images, train_labels)


"""
A single perceptron performed very well on this binary classification achieving 100%
accuracy by the end of the 7th epoch on the training data.
For fun I reduced the problem down to a smaller number of dimensions (450) using PCA
from sklearn
"""

# Dimensionality Reduction down to 450 dimensions
pca = PCA(n_components=450)
pca.fit(train_images)
pca_train_images = pca.transform(train_images)
pca_test_images = pca.transform(test_images)


# try perceptron on reduced dimensionality data
perceptron2 = Perceptron(0.01, 9, "Perceptron 2")
perceptron2.fit(pca_train_images, train_labels)
# perceptron2.graph_perceptron()

"""
Through testing I was able to determine that a reduction to 450 dimensions from the original 4200
still allows the perceptron to converge. Thus it maintains 100% accuracy on the training data.
"""

# Check Results on Test Data
results = perceptron.predict(test_images)
results_with_pca = perceptron2.predict(pca_test_images)

perceptron.calculate_results(results, test_labels, 150)
perceptron2.calculate_results(results_with_pca, test_labels, 150)

"""
Interestingly enough the perceptron with the reduced dimensions actually out performed the
perceptron without dimensional reduction. 76 correct vs 77.

The problem here is that the perceptron is suffering from over fitting. That is the model is perfect
on the training data, but that does not linearly separate the test data. For this reason I think
more significant features need to be found to get a better model.

"""

# Try PCA with 100 features
pca2 = PCA(n_components=50)
pca2.fit(train_images)
pca_train_images2 = pca.transform(train_images)
pca_test_images2 = pca.transform(test_images)

# try perceptron with 100 dimensions
perceptron3 = Perceptron(0.01, 10, "Perceptron 3")
perceptron3.fit(pca_train_images2, train_labels)
# perceptron3.graph_perceptron()
results_with_pca2 = perceptron3.predict(pca_test_images2)
perceptron3.calculate_results(results_with_pca2, test_labels, 150)

"""
Again this didn't do too well because the model is still over fitting the training data. 51.33% the same as before.
"""

# combine training and validation data to get better results
xyz = validation_data[0] + training_data[0]
abc = []
abc.extend(validation_data[1])
abc.extend(training_data[1])
print("*** Length: ", len(abc), " ***")
train_and_valid_images = np.array(xyz)
train_and_valid_labels = np.array(abc)

perceptron4 = Perceptron(0.01, 15, "Perceptron 4")
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