import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import operator
import os
import scipy.stats
from scipy.stats import norm
from statistics import mean
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from functools import reduce

desired_width=520
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# 1. Understanding and explaining the data set.

filename = "winequality-red.csv"
raw_data_file = open(filename, 'rt')
reader = csv.reader(raw_data_file, delimiter=';', quoting=csv.QUOTE_NONE)
raw_data = list(reader)
raw_data_file.close()

#print(raw_data[0]) # Specific row
#print(raw_data[0:3]) # Row 0 to 3

# Show column headers (features)
print(raw_data[0])


def get_specific_col(col_name):
    return_data = []
    col_pos = 0
    column_headers = raw_data[0]
    raw_data_no_header = raw_data[1:len(raw_data)]
    for k in range(len(column_headers)):
        if column_headers[k].strip('\"') == col_name:
            col_pos = k
            break

    for i in range(len(raw_data_no_header)):
        return_data.append(float(raw_data_no_header[i][col_pos]))

    return return_data

## Basic Statistical Info
# Which feature do we want?
def view_data_stat_info(raw_data_feature_name):
    raw_data_feature_list = get_specific_col(raw_data_feature_name)
    print("Mean: " + str(np.mean(raw_data_feature_list)))
    print("Median: " + str(np.median(raw_data_feature_list)))
    print("fixed_acidity, sorted: " + str(np.sort(raw_data_feature_list)))
    print("Max: " + str(np.max(raw_data_feature_list)))
    print("Min: " + str(np.min(raw_data_feature_list)))
    print("Std Deviation: " + str(np.std(raw_data_feature_list, ddof=1)))
    #np.std([0, 1], ddof=1)
    # Include standard deviation, IQR

print(view_data_stat_info("fixed acidity"))


######## DATA VISUALIZATION
raw_data_feature_list = get_specific_col("fixed acidity")
x_for_plot = np.arange(0, len(raw_data_feature_list))
#y_for_plot = 2 * x + 5
y_for_plot = np.sort(raw_data_feature_list)
y_for_plot = raw_data_feature_list

plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("target data_feature")
plt.plot(x_for_plot, y_for_plot)
plt.show()


plt.scatter(x_for_plot, y_for_plot)
plt.show()



plt.title("Target Data Feature Histogram")
plt.xlabel("target data feature")
plt.ylabel("Occurrences")
plt.hist(raw_data_feature_list, bins=10)
#plt.show()


# Plot the PDF.
# https://www.science-emergence.com/Articles/How-to-plot-a-normal-distribution-with-matplotlib-in-python-/
x_min = np.min(raw_data_feature_list)
x_max = np.max(raw_data_feature_list)

mean = np.average(raw_data_feature_list)
std = np.std(raw_data_feature_list)
x = np.linspace(x_min, x_max, 100)
y = scipy.stats.norm.pdf(x, mean, std)
plt.plot(x, y)
plt.grid()
plt.xlim(x_min, x_max)
plt.ylim(0, 0.50)
plt.title('Distribution of Data', fontsize=10)
plt.xlabel('x')
plt.ylabel('Normal Distribution')
# plt.savefig("normal_distribution.png")
plt.show()



# Using Dataframe - NOT USED
df = pd.DataFrame(raw_data)
#  columns = ["a", "b", "c", "d", "d", "d", "d", "d", "f", "f", "f", "f"]
header = df.iloc[0]
df = df[1:]
df.columns = header




# 2. Processing data, cleaning up.
# change semicolon to comma
filename = "wine-comma.csv"
if os.path.isfile(filename) == True:
    os.remove(filename)

for i in range(len(raw_data)):
    temp_str = str(raw_data[i])
    data_csv = temp_str.replace(";", ",")
    data_csv = data_csv.replace("\"", "")
    data_csv = data_csv.replace("\'", "")
    data_csv = data_csv.replace("[", "")
    data_csv = data_csv.replace("]", "")
    data_csv = data_csv.replace(" ", "")
    file = open('wine-comma.csv', 'a')
    file.write(data_csv+"\n")
    file.close()
raw_data_clean = open(filename, 'rt')
reader = csv.reader(raw_data_clean, delimiter=',', quoting=csv.QUOTE_NONE)
data_clean = list(reader)
raw_data_clean.close()
#print(data_clean)


data_pd = pd.read_csv('wine-comma.csv')

# 3. Dividing your data into a training and test set.

# Identify features to predict labels. In this case. Let's try quality.
target_feature = "quality"  # The feature to be predicted

data_pd_y = data_pd[target_feature]
data_pd_x = data_pd.drop(target_feature, axis=1)
x_train, x_test, y_train, y_test = train_test_split(data_pd_x, data_pd_y, train_size=0.80, test_size=0.20)


# 4. Choosing the relevant algorithm.
# Text write up in Jupyter


# 5. Writing a python code to perform learning. (You can reuse every code from the lectures)
# KNN fit
def knn_lib_fit(x_train, y_train):
    knn_x = x_train
    knn_y = y_train
    knn = KNeighborsClassifier(n_neighbors=3)
    try:
        knn.fit(knn_x, knn_y)
    except Exception as error:
        print('Error: ' + repr(error))
        if repr(error) == "ValueError(\"Unknown label type: 'continuous'\")":
            print("erropr here ==================")
            # It is float column. So convert.
            knn_y = knn_y.apply(lambda x: int(x * 100000))
            #print(log_reg_y)
            knn.fit(knn_x, knn_y)

    y_pred_knn = knn.predict(knn_x)
    return y_pred_knn

print("KNN Lib Fit:")
knn_lib_prediction = knn_lib_fit(x_train, y_train)
print("knn_lib_prediction: " + str(knn_lib_prediction))


# Regression - Plot to see predictions
model = lm().fit(x_train, y_train)
predictions = model.predict(x_test)
plt.scatter(y_test, predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()


# Log Regression fit
def log_reg_fit(x_train, y_train):
    log_reg_x = x_train
    log_reg_y = y_train
    logreg = LogisticRegression(solver='liblinear', multi_class='auto')
    try:
        logreg.fit(log_reg_x, log_reg_y)
    except Exception as error:
        if repr(error) == "ValueError(\"Unknown label type: 'continuous'\")":
            # It is float column. So convert.
            log_reg_y = log_reg_y.apply(lambda x: int(x * 100000))
            logreg.fit(log_reg_x, log_reg_y)

    y_pred_log_reg = logreg.predict(log_reg_x)
    return y_pred_log_reg


print("log_reg_fit prediction:")
print(log_reg_fit(x_train, y_train))

# KNN without using libraries
def createDataSet(target_feature_to_create_dataset, source_features_list):
    labels = data_pd[target_feature_to_create_dataset]
    # labels = data_pd_y.values.tolist() #This is the list of target feature. ex. Quality.
    group = []  # This will be the features selected.
    for i in range(len(data_pd)):
        wine_selected_features_row = []

        for j in range(len(source_features_list)):
            wine_selected_features_row.append(data_pd[source_features_list[j]].values.tolist()[i])

        group.append(wine_selected_features_row)
    return group, labels



def knn_classifier_no_lib(inX, dataSet, labels, k):
    dataSetSize = len(dataSet)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

source_features_list_for_create_dataset = ["fixedacidity", "volatileacidity", "alcohol"]
group, labels = createDataSet("quality", source_features_list_for_create_dataset)

# Run it to test with specific values.
knn_no_lib_prediction = knn_classifier_no_lib([11.2, 0.28, 9.8], group, labels, 3)
print(knn_no_lib_prediction)


# 6. Evaluating your learning performance.

# Accuracy measure for log reg
def log_reg_fit_accuracy(x_train, y_train):
    y_pred_log_reg = log_reg_fit(x_train, y_train)
    log_reg_y = y_train
    log_reg_accuracy_score = "NULL"
    try:
        log_reg_accuracy_score = metrics.accuracy_score(log_reg_y, y_pred_log_reg)
    except Exception as error:
        print(repr(error))
        if repr(error) == "ValueError(\"Classification metrics can't handle a mix of continuous and multiclass targets\")":
            # It is float column. So convert.
            log_reg_y = log_reg_y.apply(lambda x: int(x * 100000))
            log_reg_accuracy_score = metrics.accuracy_score(log_reg_y, y_pred_log_reg)

    return log_reg_accuracy_score

print("Log Regression Prediction Accuracy:")
log_reg_accuracy_score = log_reg_fit_accuracy(x_train, y_train)
print(log_reg_accuracy_score)

# Accuracy measure for KNN
print("KNN Predictions Accuracy:")
#print(metrics.accuracy_score(knn_y,y_pred_knn))
knn_lib_prediction = knn_lib_fit(x_train, y_train)
print(metrics.accuracy_score(y_train, knn_lib_prediction))

# Measure for KNN no lib
def eval_knn_no_lib_classifier(target_feature_for_eval, source_features_list):
    wrong_prediction = 0
    group, labels = createDataSet(target_feature_for_eval, source_features_list)

    for i in range(len(data_pd)):
        source_features_list_row_value = []
        for j in range(len(source_features_list)):
            source_features_list_row_value.append(data_pd[source_features_list[j]].values.tolist()[i])

        target_feature_for_comparing = data_pd[target_feature_for_eval][i]

        # Line below is the classifier we're evaluating
        knn_no_lib_prediction = knn_classifier_no_lib(source_features_list_row_value, group, labels, 3)

        if knn_no_lib_prediction != target_feature_for_comparing:
            wrong_prediction = wrong_prediction + 1
    error_rate = wrong_prediction / len(data_pd.quality)
    return error_rate


source_features_list_for_create_dataset = ["fixedacidity", "volatileacidity", "pH"]
knn_classifier_error_rate = eval_knn_no_lib_classifier("pH", source_features_list_for_create_dataset)
print("knn_classifier_error_rate: (1.0 means classifier is always wrong. The lower the better.)")
print(knn_classifier_error_rate)




# 7. Making sure your results does not depend on your choosing parameters.

# Log reg accuracy
print("Log Regression Prediction Accuracy for x feature:")
log_reg_accuracy_score = log_reg_fit_accuracy(x_train, y_train)
print(log_reg_accuracy_score)

# Accuracy measure for KNN
print("KNN Predictions Accuracy:")
knn_lib_prediction = knn_lib_fit(x_train, y_train)
print(metrics.accuracy_score(y_train, knn_lib_prediction))

# Measure for KNN no lib
# Using a different target. In this case "alcohol"

# Here's a function the loops through the entire feature set

def eval_all_feature_error_rate(source_features_list):
    for i in range(len(data_pd.columns.values)):
        knn_classifier_error_rate = eval_knn_no_lib_classifier(data_pd.columns.values[i], source_features_list)
        print("Data Feature: " + str(data_pd.columns.values[i]) + ", Error Rate: " + str(knn_classifier_error_rate))

source_features_list_for_all_feature_error_rate = ["citricacid", "residualsugar", "freesulfurdioxide"]
eval_all_feature_error_rate(source_features_list_for_all_feature_error_rate)

