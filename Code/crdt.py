# import necessary packages
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from pylab import rcParams

rcParams['figure.figsize'] = 14,8
RANDOM_SPEED = 42
LABELS = ["Normal", "Fraud"]

# load dataset from csv file using pandas
data = pd.read_csv('creditcard.csv', sep=',')

#grab a peek at the data
print(data.head())

#information about the dataset
data.info()

#check if there is any null value
print("null", data.isnull().values.any())

#print the shape of the data
print(data.shape)
print(data.describe())

#how many number of classes are there
count_classes = pd.value_counts(data['Class'], sort=True)
count_classes.plot(kind = 'bar', rot=0)

#frequency of classes
plt.title("transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")

#get the fraud and the normal datset
fraud = data[data['Class']==1]
normal = data[data['Class']==0]

#print the fraud and normal dataset
print(fraud.shape, normal.shape)

#How different are the amount of money used in different transaction classes
fraud.Amount.describe()
normal.Amount.describe()

#amount per transection by class
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show()

#do fraudulent transactions occur more often during certain time frame
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

# Take some sample of the data

data1= data.sample(frac = 0.1,random_state=1)

data1.shape
data.shape

#Determine the number of fraud and valid transactions in the dataset

Fraud = data1[data1['Class']==1]

Valid = data1[data1['Class']==0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Valid)))

# loading data
data = pd.read_csv('creditcard.csv', sep=',')
print(data.shape)

# COllecting labels
label = data.Class.values
print(label)

# Removing labels from dataset
data.drop('Class', axis=1, inplace=True)
print(data.head())

# Test - Train split (80-20)
data_train, data_test, labels_train, labels_test = train_test_split(data, label, random_state=0)

print('Train Data', data_train.shape)
print('Test Data', data_test.shape)

# Model Training
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data_train, labels_train)

# Predicting
train_predict = knn.predict(data_train)
test_predict = knn.predict(data_test)

# Accuracy scores
train_accuracy = accuracy_score(labels_train, train_predict)
test_accuracy = accuracy_score(labels_test, test_predict)

print('Training Accuracy', train_accuracy)
print('Test Accuracy', test_accuracy)