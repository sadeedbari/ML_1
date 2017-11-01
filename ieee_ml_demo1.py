""" Basic classification project on
    Iris flower data set
"""

#importing libraries
import pandas
from pandas.tools.plotting import scatter_matrix
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#loading data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#shape
print(dataset.shape)
print("\n")

#head (peeking at the top of the data)  -- commented out
print(dataset.head(20))

#descriptions of each attribute and other useful statistics
print(dataset.describe())
print("\n")

#class distribution (aka how many instances (rows) belong to each class)
print(dataset.groupby('class').size())
print("\n")

#Graphing the data
#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

#historgrams
dataset.hist()
plt.show()

#scatter plot matrix
scatter_matrix(dataset)
plt.show()
