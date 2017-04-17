import load_face_data
from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

training_data, validation_data, test_data = load_face_data.load_data()

print(len(training_data[0]))
print(len(training_data[1]))

x = training_data[0]
y = training_data[1]



train_images = np.array(training_data[0])
train_labels = np.array(training_data[1])

perceptron = Perceptron(0.01, 10)
perceptron.fit(train_images, train_labels)
plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications")
plt.show()


"""
A single perceptron performed very well on this binary classification achieving 100%
accuracy by the end of the 7th epoch.
For fun I reduced the problem down to a smaller number of dimensions using PCA
from sklearn
"""

print("==================================================================")

# Dimensionality Reduction down to 450 dimensions
pca = PCA(n_components=450)
pca.fit(train_images)
train_images = pca.transform(train_images)


# try perceptron on reduced dimensionality data
perceptron2 = Perceptron(0.01, 10)
perceptron2.fit(train_images, train_labels)
plt.plot(range(1, len(perceptron2.errors) + 1), perceptron2.errors, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Number of misclassifications 2")
plt.show()

"""
Through testing I was able to determine that a reduction to 450 dimensions from the original 4500
still allows the perceptron to converge. Thus it maintains 100% accuracy
"""