import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Sample data
X = np.array([
    [2, 3], [3, 4], [4, 5], [2.5, 4], [3.5, 5], [3.5, 4],  # Class 1
    [7, 8], [8, 9], [9, 10], [7.5, 9], [8.5, 10], [9.5, 8.5]  # Class 2
])

y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # Labels

# Create and train the SVM model
model = svm.SVC(kernel='linear')
model.fit(X, y)

# Get the separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0.5, 9.5)
yy = a * xx - (model.intercept_[0]) / w[1]

# Plotting the points and the hyperplane
plt.scatter(X[:6, 0], X[:6, 1], c='red', label='Spam')
plt.scatter(X[6:, 0], X[6:, 1], c='blue', label='Not Spam')
#plt.scatter(A[:, 0], A[:, 1], c='black', label='New Emails')
#plt.scatter(X[])
plt.xlim(0, 10)
plt.ylim(2, 12)
plt.plot(xx, yy, 'k-', label='Decision Boundary')

# # Plot support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k', label='Support Vectors')

plt.title('SVM Classification')
plt.xlabel('Length of the Email')
plt.ylabel('Frequency of word "Free"')
plt.legend(loc='upper center')
plt.show()
