#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import math

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# features_list created further at bottom
# similar potential_features variable created when new features created below

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
print("Length of dict: ", len(data_dict))

### Task 2: Remove outliers
# Outliers identified by exploring dataset with pandas
# Removes keys corresponding to outliers from data_dict
data_dict = {k:v for k,v in data_dict.items() if k not in ['TOTAL', 
    'LOCKHART EUGENE E', 'THE TRAVEL AGENCY IN THE PARK']}

### Task 3: Create new feature(s)
def make_new_feature(feature_name, input1, input2):
    if v[input1] == 'NaN' or v[input2] == 'NaN':
        v[feature_name] = 'NaN'
    else:
        v[feature_name] = int(v[input1]) / int(v[input2])

for k, v in data_dict.items():
    make_new_feature('prop_of_total_payments_was_bonus', 'bonus', 'total_payments')
    make_new_feature('prop_emails_from_poi_to_person', 'from_poi_to_this_person', 'to_messages')
    make_new_feature('prop_emails_to_poi_from_person', 'from_this_person_to_poi', 'from_messages')

# New list of potential features, with the newly-created features added
potential_features = ['poi', 'bonus', 'salary', 'to_messages', 'deferral_payments',
    'total_payments', 'restricted_stock_deferred', 'deferred_income',
    'total_stock_value', 'expenses', 'exercised_stock_options',
    'from_messages', 'other', 'long_term_incentive',
    'other', 'from_this_person_to_poi', 'long_term_incentive',
    'shared_receipt_with_poi', 'restricted_stock', 'director_fees',
    'prop_emails_from_poi_to_person', 'prop_emails_to_poi_from_person', 
    'prop_of_total_payments_was_bonus']

# Final features selected using below algorithm
final_features = ['poi', 'bonus', 'prop_of_total_payments_was_bonus',
                  'exercised_stock_options', 'total_stock_value']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, potential_features, sort_keys = True)
labels, features = targetFeatureSplit(data)
# Comment the above lins after feature selection is done
# Uncomment the below if using to validate final features (i.e. not selecting features)
# data = featureFormat(my_dataset, final_features, sort_keys = True)
# labels, features = targetFeatureSplit(data)

# get basic information on dataset
from collections import Counter
print("Basic stats:")
print("Number of data points: ", len(labels))
print("Number of pois: ", Counter(labels))

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score


# scale features
scaler = MinMaxScaler()
scaler.fit(features)
features_scaled = scaler.transform(features)

# initiates naive bayes classifier
clf = GaussianNB()

# removes poi from potential_features so its index matches up with the feature scores
# this is necessary for below step to get feature information
potential_features.remove('poi')

# perform validation for shuffled, stratified cross-sections of dataset
precision_scores = []
recall_scores = []
new_features = {}

sss = StratifiedShuffleSplit(n_splits=5, random_state=42)
for train_index, test_index in sss.split(features_scaled, labels):

    # prepare training and testing data for this cross-section
    features_train, features_test = features_scaled[train_index], features_scaled[test_index]
    labels = np.asarray(labels)
    labels_train, labels_test = labels[train_index], labels[test_index]

    # select k best features (comment out if not using feature selection)
    selector = SelectKBest(chi2, k=2)
    selector.fit_transform(features_train, labels_train)
    features_train = selector.transform(features_train)
    features_test = selector.transform(features_test)
    
    # get information on features used in this cross-section (comment out after feature selection)
    mask = selector.get_support()
    for bool, feature, chi2_score in zip(mask, potential_features, selector.scores_):
        if bool:
            if feature not in new_features:
                new_features[feature] = [chi2_score]
            else:
                new_features[feature].append(chi2_score)

    # fit GaussianNB model and get recall and precision score for cross-section
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    precision_scores.append(precision_score(labels_test, pred))
    recall_scores.append(recall_score(labels_test, pred))

# can comment out if not using feature selection; otherwise there will be no values
print("Features used in model:")
for k, v in new_features.iteritems():
    print("Feature: ", k)
    print("Number of times feature was one of K best: ", len(new_features[k]))
    print("Mean chi2_score: ", np.mean(new_features[k]))

print('Precision: ', np.mean(precision_scores))    
print('Recall: ', np.mean(recall_scores))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

features_list = ['poi', 'prop_of_total_payments_was_bonus', 'exercised_stock_options']

dump_classifier_and_data(clf, my_dataset, features_list)
