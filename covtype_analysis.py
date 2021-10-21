from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, recall_score, \
    roc_auc_score, precision_score, average_precision_score
from keras.losses import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import time as time


# Evaluation function
training_times = []
testing_times = []
def model_evaluation(clf):
    # passing classifier to a variable
    clf = clf

    # records time
    t_start = time.time()
    # classifier learning the model
    clf = clf.fit(X_train, y_train)
    # records time
    t_end = time.time()

    # records time
    c_start = time.time()
    # make predictions
    y_pred = clf.predict(X_test)

    precision = precision_recall_fscore_support(y_test, y_pred, average='macro')

    # records time
    c_end = time.time()

    # substracts end time with start to give actual time taken in seconds
    # divides by 60 to convert in minutes and rounds the answer to three decimal places
    # time in training
    t_time = np.round((t_end - t_start) / 60, 3)
    training_times.append(t_time)
    # time for evaluating scores
    c_time = np.round((c_end - c_start) / 60, 3)
    testing_times.append(c_time)

    # returns performance measure and time of the classifier
    # print("Classifier : ", clf ,"\tAccuracy :", acc_mean, "\tRecall", recall_mean ,"% \tF1-score", f1_mean)
    print("Classifier : ", clf, "\t Accuracy :", round(precision[0] * 100,2), "\tRecall", round(precision[1] * 100,2), "% \tF1-score", round(precision[2] * 100, 2),
          "% taking", t_time, "minutes to train and", c_time, "minutes to evaluate metric scores..")

    # Removing traces of classifier
    clf = None


# First 15120 rows for the training set
def create_datasets(data):
    X = data.iloc[:, :-1]  # Features - All columns but last

    X_train = X[:15120].copy()
    # The last seven colums are the targets
    X_train, y_train = X_train.iloc[:, :51], X_train.iloc[:, -1]
    # The remaining rows are for the test set
    X_test = X[15120:].copy()
    X_test, y_test = X_test.iloc[:, :51], X_test.iloc[:,-1]
    return X_train, X_test, y_train, y_test

data = pd.read_csv('covtype.data', delimiter=",")
print(data.shape)

# Data analysis
# Check if there are null values in the data
print(data.isna().sum())

# Define target names, since the data doesn't have column names, we will provide it in a form of list
target_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2',
                'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7',
                'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
                 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
                 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
                 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type']

# Insert target names as columns names
curr_col_names = list(data.columns)

mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = target_names[i]

data = data.rename(columns=mapper)
print(data.head())

X_train, X_test, y_train, y_test = create_datasets(data)

#Standardizing the data i.e. to rescale the features to have a mean of zero and standard deviation of 1.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape)

print("[INFO] Building, training, testing and comparing the models")
# Build the logistic regression model and models to be compared against
first_log = LogisticRegression(penalty='l2', tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1, solver='lbfgs', max_iter=10000, multi_class='auto', verbose=1, warm_start=False, l1_ratio=None, random_state=42)
RFC = RandomForestClassifier(random_state = 42)
Ada = AdaBoostClassifier()
GBC = GradientBoostingClassifier(random_state = 42)

# Initiate list of models
models = [first_log, RFC, Ada, GBC]

# Integrate second logistic regression pipeline with different parameter values
tols = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
first_mean_acc = []
first_general = []
for i in range(0,len(tols)):
    # liblinear solver supported for the l1 penalty
    sec_log = LogisticRegression(penalty='l1', tol=tols[i], C=1.0, fit_intercept=True, intercept_scaling=1,
                           solver='liblinear', max_iter=1000, multi_class='auto', verbose=1, warm_start=False, l1_ratio=None)

    sec_log.fit(X_train, y_train)
    y_pred_sec_log_1 = sec_log.predict(X_test)

    # Store Performance(based on accuracy) report
    first_mean_acc.append(y_pred_sec_log_1)

    met = precision_recall_fscore_support(y_test, y_pred_sec_log_1, average='macro')
    first_general.append(met)


# Integrate second logistic regression pipeline with different parameter values
iters = [10, 100, 1000, 5000, 10000]
second_mean_acc = []
second_general = []
for k in range(0,len(iters)):
    # saga solver supported for the l1 penalty
    sec_log = LogisticRegression(penalty='l1', tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1,
                           solver='saga', max_iter=iters[k], multi_class='auto', verbose=1, warm_start=False, l1_ratio=None)

    sec_log.fit(X_train, y_train)
    y_pred_sec_log_2 = sec_log.predict(X_test)

    # Store Performance(based on accuracy) report
    second_mean_acc.append(y_pred_sec_log_2)

    meta = precision_recall_fscore_support(y_test, y_pred_sec_log_2, average='macro')
    second_general.append(meta)

'''''
# # Performance(based on accuracy) report
# print('Logistic Regression Accuracy: %.3f (+/- %.3f)' % (np.mean(y_pred_first_log), np.std(y_pred_first_log)))
# print('Linear Regression Accuracy: %.3f (+/- %.3f)' % (np.mean(y_pred_lr), np.std(y_pred_lr)))
# print('Lasso Accuracy: % .3f(+ / - % .3f)' % (np.mean(y_pred_lasso), np.std(y_pred_lasso)))
# print('Ridge Accuracy: %.3f (+/- %.3f)' % (np.mean(y_pred_ridge), np.std(y_pred_ridge)))
#
# print('Logistic Regression Accuracy: ', first_log.score(X_test,y_test))
# print('Linear Regression Accuracy: ', lr.score(X_test,y_test))
# print('Lasso Accuracy: ', lasso.score(X_test,y_test))
# print('Ridge Accuracy: ', ridge.score(X_test,y_test))

# for j in range(0,len(mean_acc)):
#     #np.int(mean_acc[j])
#     mean_acc = np.vectorize(mean_acc[j])
#     print('Second Logistic Regression Accuracy[L1, solver=liblinear] : ', mean_acc[j])
'''''


for index in range(0, len(models)):
    # passing the model to function to get performance measures
    model_evaluation(models[index])


#print('Second Logistic Regression Experiment Accuracy: %.3f (+/- %.3f)' % (np.mean(first_mean_acc), np.std(first_mean_acc)))
print("[INFO REPORT] \t\t\t\t\t\t\t\t\t\t\t\t Precision \t\t\t\t Recall \t\t\t F1-Score \t Support ")
for len in range(0,len(first_general)):
    print('Second Logistic Regression scores[based on tolerance] : ', first_general[len])


#print('Thirtd Logistic Regression Experiment Accuracy: %.3f (+/- %.3f)' % (np.mean(second_mean_acc), np.std(second_mean_acc)))
print("[INFO REPORT] \t\t\t\t\t\t\t\t\t\t\t Precision \t\t\t\t Recall \t\t\t F1-Score \t Support ")
for number in range(5):
    print('Third Logistic Regression scores[based on iters] : ', second_general[number])


# LDA
lda = LinearDiscriminantAnalysis()

new_t_start = time.time()

results = lda.fit(X_train, y_train)

new_t_end = time.time()

new_tp_start = time.time()

lda_y_predicted = lda.predict(X_test)

lda_train_accuracy = lda.score(X_train, y_train)
lda_accuracy = lda.score(X_test, y_test)

new_tp_end = time.time()

# Calculating actuall times
lda_train_time = np.round((new_t_end - new_t_start) / 60, 3)
lda_test_time = np.round((new_tp_end - new_tp_start) / 60, 3)


print("[TRAINING INFO]")
print("Training Accuracu - LDA: ", lda_train_accuracy)

print("[TESTING INFO]")
print('Accuracy Score - LDA:', accuracy_score(y_test, lda_y_predicted))
print('Average Precision - LDA:',average_precision_score(y_test, lda_y_predicted))
print('F1 Score - LDA:',f1_score(y_test, lda_y_predicted))
print('Precision - LDA:', precision_score(y_test, lda_y_predicted))
print('Recall - LDA:', recall_score(y_test, lda_y_predicted))
print('ROC Score - LDA:', roc_auc_score(y_test, lda_y_predicted))

print("Convergence Time Comparison")
print("\t\t\t Training  |  Testing")
print("LDA : \t\t\t{} \t\t{} ".format(lda_train_time, lda_test_time))


for mt in range(4):
    print(models[mt], ": \t\t\t{} \t\t{} ".format(training_times[mt], testing_times[mt]))
