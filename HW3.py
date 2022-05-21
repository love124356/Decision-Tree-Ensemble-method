# HW3: Decision Tree, AdaBoost and Random Forest
# In hw3, you need to implement decision tree,
# adaboost and random forest by using only numpy,
# then train your implemented model by the provided dataset
# and test the performance with testing data

# Please note that only **NUMPY** can be used to implement your model,
# you will get no points by simply calling sklearn.tree.DecisionTreeClassifier

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score

# Load data
# The dataset is the Heart Disease Data Set from UCI Machine Learning Repository.
# It is a binary classifiation dataset, the label is stored in `target` column.
# See follow links for more information: https://archive.ics.uci.edu/ml/datasets/heart+Disease

file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
df = pd.read_csv(file_url)

train_idx = np.load('train_idx.npy')
test_idx = np.load('test_idx.npy')

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

train_df = train_df.replace({'fixed': 0, 'reversible': 1, 'normal': 2})
test_df = test_df.replace({'fixed': 0, 'reversible': 1, 'normal': 2})

X_train = train_df.iloc[:, :-1].values
Y_train = train_df.iloc[:, -1].values.reshape(-1, 1)

X_test = test_df.iloc[:, :-1].values
Y_test = test_df.iloc[:, -1].values.reshape(-1, 1)

# ## Question 1
# Please compute the Entropy and Gini Index of provided data.
# Ref.: https://reurl.cc/p1xo7a


def gini(sequence):
    n = len(sequence)
    _, count = np.unique(sequence, return_counts=True)
    prob = count / n

    g = 1 - np.sum(prob ** 2)
    return g


def entropy(sequence):
    n = len(sequence)
    _, count = np.unique(sequence, return_counts=True)
    prob = count / n

    e = -1 * np.sum(prob * np.log2(prob))
    return e


# 1 = class 1,
# 2 = class 2
data = np.array([1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2])

print("Gini of data is ", gini(data))
print("Entropy of data is ", entropy(data))
print("=============================================")


# ## Question 2

# Implement the Decision Tree algorithm (CART, Classification and Regression Trees)
# and trained the model by the given arguments, and print the accuracy score on the test data.
# You should implement two arguments for the Decision Tree algorithm
# 1. **criterion**: The function to measure the quality of a split.
# Your model should support `gini` for the Gini impurity and `entropy` for the information gain.
# 2. **max_depth**: The maximum depth of the tree. If `max_depth=None`,
# then nodes are expanded until all leaves are pure. `max_depth=1` equals to split data once

# - Note: All of your accuracy scores should over **0.7**
# - Note: You should get the same results when re-building the model with the same arguments,
#         no need to prune the trees
# - Hint: You can use the recursive method to build the nodes
# Ref.: https://reurl.cc/1ZoMDW

class Node():
    """Build tree nodes.

    For recording some information of each decision tree node.

    Parameters
    ----------
    feature_index : int, default=None
        which value is used.

    threshold : int or float, default=None
        The best split threshold of the tree node.

    left : Node, default=None
        The left sub-tree of this node.

    right : Node, default=None
        The right sub-tree of this node.

    quality: int or float, default=None
        The quality of this node.

    value: int, default=None
        The y-value of this leaf node. Only for leaf node.
    """
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        quality=None,
        value=None
    ):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.quality = quality

        # for leaf node
        self.value = value


class DecisionTree():
    """A decision tree classifier.

    Decision Trees (DTs) are a non-parametric supervised learning method used for classification.
    The goal is to create a model that predicts the value of a target variable
    by learning simple decision rules inferred from the data features.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure.

    max_features : int or float, default=None
        The number of features to consider when looking for the best split.

    Attributes
    ----------
    total_feature_num : int, default=None
        The total number of features.

    total_use_feature : list, default=None
        The total number of times that this feature is used to split.
    """
    def __init__(self, criterion='gini', max_depth=None, max_features=None):
        ''' constructor '''

        # initialize the root of the tree
        self.root = None
        # initialize the parameters
        self.depth = 0
        self.criterion = criterion
        self.max_features = max_features
        # record total feature number
        self.total_feature_num = None
        # for feature importance
        self.total_use_feature = None

        # stopping conditions
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=1):
        ''' recursive function to build the tree '''

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # For random forest. Replace the num_features to max_features
        if self.max_features is not None:
            num_features = self.max_features

        if self.max_depth is None:
            # Set to the number of samples to ensure that each leaf node can be pure
            # because it will be judged later whether node is pure.
            self.max_depth = num_samples

        # split until stopping conditions are met
        if curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_features)
            # print(best_split)

            # check whether best_split is empty and quality is not pure
            if len(best_split) != 0:
                # if node is pure, it will automatically change to leaf node
                if best_split["quality"] > 0:
                    # recursive left
                    left_subtree = self.build_tree(
                        best_split["left"], curr_depth+1)
                    # recursive right
                    right_subtree = self.build_tree(
                        best_split["right"], curr_depth+1)
                    # return decision node
                    return Node(
                        best_split["feature_index"], best_split["threshold"],
                        left_subtree, right_subtree, best_split["quality"]
                    )

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_features):
        ''' function to find the best split '''

        # dictionary to store the best split
        best_split = {}
        max_quality = -float("inf")

        if num_features == self.total_feature_num:
            # For the same decision result constraint
            # Fix the order of feature
            feature_idx_list = [i for i in range(self.total_feature_num)]
        else:
            # For random forest
            feature_idx_list = random.sample(
                range(self.total_feature_num), num_features)

        # loop over all the features
        for feature_index in feature_idx_list:
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # Use the i-th threshold is the average of the i-th and (i+1)-th sorted values
            thresholds_conditions = []
            for t in range(len(possible_thresholds)-1):
                newT = (possible_thresholds[t] + possible_thresholds[t]) / 2
                thresholds_conditions.append(newT)

            # loop over all the feature values present in the data
            # Optional: Use the unique sorted value of the feature as the threshold to split
            # for threshold in possible_thresholds:
            for threshold in thresholds_conditions:
                # get current split
                left, right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(left) > 0 and len(right) > 0:
                    y = dataset[:, -1]
                    left_y, right_y = left[:, -1], right[:, -1]
                    # compute the quality
                    current_quality = self.information_gain(
                        y, left_y, right_y, self.criterion)
                    # update the best split if needed
                    if current_quality > max_quality:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left"] = left
                        best_split["right"] = right
                        best_split["quality"] = current_quality
                        max_quality = current_quality

        # return best split
        return best_split

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''

        left = np.array(
            [row for row in dataset if row[feature_index] <= threshold]
        )
        right = np.array(
            [row for row in dataset if row[feature_index] > threshold]
        )
        return left, right

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute the quality '''

        # each child weight
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (
                weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child)
            )
        else:
            gain = self.entropy(parent) - (
                weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child)
            )

        return gain

    def entropy(self, y):
        ''' function to compute entropy '''

        n = len(y)
        _, count = np.unique(y, return_counts=True)
        prob = count / n

        e = -1 * np.sum(prob * np.log2(prob))
        return e

    def gini_index(self, y):
        ''' function to compute gini index '''

        n = len(y)
        _, count = np.unique(y, return_counts=True)
        prob = count / n

        g = 1 - np.sum(prob ** 2)
        return g

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''

        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index),
                  "<=", tree.threshold, "?", tree.quality)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        ''' function to train the tree '''

        self.total_feature_num = X_train.shape[1]
        dataset = np.concatenate((X, Y), axis=1)
        # print(dataset)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        ''' function to predict new dataset '''

        preditions = [self.make_prediction(v, self.root) for v in X]
        return preditions

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''

        if tree.value is not None:
            return tree.value

        feature_val = x[tree.feature_index]

        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def get_feature(self, node):
        ''' go through the whole tree and record the feature that using in each node '''

        if node.left and node.left.quality is not None:
            self.get_feature(node.left)
        if node.right and node.right.quality is not None:
            self.get_feature(node.right)
        # calculate the total times that feature used
        self.total_use_feature[node.feature_index] += 1

    def feature_importance(self):
        ''' calculate the feature importance of the all the features '''
        self.total_use_feature = np.zeros(self.total_feature_num)
        self.get_feature(self.root)

        return self.total_use_feature

# ### Question 2.1
# Using `criterion=gini`, showing the accuracy score of test data
# by `max_depth=3` and `max_depth=10`, respectively.


clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
clf_depth3.fit(X_train, Y_train)
# clf_depth3.print_tree()
Y_pred = clf_depth3.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("clf_depth3: ", acc)

clf_depth10 = DecisionTree(criterion='gini', max_depth=10)
clf_depth10.fit(X_train, Y_train)
# clf_depth10.print_tree()
Y_pred = clf_depth10.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("clf_depth10: ", acc)

print("\nTest begin===================================")
# Test max_depth is one and None
clf_depthNone = DecisionTree(criterion='gini', max_depth=None)
clf_depthNone.fit(X_train, Y_train)
# clf_depthNone.print_tree()
Y_pred = clf_depthNone.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("clf_depthNone: ", acc)


clf_depth1 = DecisionTree(criterion='gini', max_depth=1)
clf_depth1.fit(X_train, Y_train)
# clf_depth1.print_tree()
Y_pred = clf_depth1.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("clf_depth1: ", acc)
print("=====================================Test end\n")


# ### Question 2.2
# Using `max_depth=3`, showing the accuracy score of test data
# by `criterion=gini` and `criterion=entropy`, respectively.

clf_gini = DecisionTree(criterion='gini', max_depth=3)
clf_gini.fit(X_train, Y_train)
# clf_gini.print_tree()
Y_pred = clf_gini.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("clf_gini: ", acc)

clf_entropy = DecisionTree(criterion='entropy', max_depth=3)
clf_entropy.fit(X_train, Y_train)
# clf_entropy.print_tree()
Y_pred = clf_entropy.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("clf_entropy: ", acc)
print("=============================================")


# ## Question 3
# Plot the feature importance of your Decision Tree model.
# You can get the feature importance by counting the feature used for splitting data.

# - You can simply plot the **counts of feature used** for building tree without normalize the importance.
# Take the figure below as example, outlook feature has been used for splitting for almost 50 times.
# Therefore, it has the largest importance
# [image](https://i2.wp.com/sefiks.com/wp-content/uploads/2020/04/c45-fi-results.jpg?w=481&ssl=1)

# Ref.: https://reurl.cc/anNzOG

f = clf_depth10.feature_importance()
feature_names = list(df.columns)
feature_names.pop()  # Pop the "target column"

plt.figure(figsize=(8, 6))
plt.barh(feature_names, f, color='#278AA8')
plt.title("Feature Importance")
plt.ylabel('feature names')
plt.xlabel('counts')
plt.yticks(feature_names, feature_names)
plt.grid(True)
plt.savefig('feature_importance.png', dpi=200)
# plt.show()

# ## Question 4
# implement the AdaBooest algorithm by using the CART you just implemented from question 2 as base learner.
# You should implement one arguments for the AdaBooest.
# 1. **n_estimators**: The maximum number of estimators at which boosting is terminated

# Ref.: https://reurl.cc/j1gpLL, https://reurl.cc/QLj30O


class AdaBoost():
    """An AdaBoost classifier.

    Boosting method that uses a number of weak classifiers in
    ensemble to make a strong classifier. This implementation uses decision
    stumps, which is a one level Decision Tree (with depth one).

    Parameters:
    -----------
    n_estimators: int, default=10
        The number of weak classifiers that will be used.

    random_state: int, default=53
        The seed for sampling.

    Attributes:
    -----------
    predictor_weight: object
        The weights of predictors based on its error. Record which tree, index and its weights.

    X_train: array
        Use the original training data when sampling.

    y_train: array
        Use the original training labels when sampling.
    """
    def __init__(self, n_estimators=10, random_state=53):
        ''' constructor '''
        # initialize the parameters
        self.n_estimators = n_estimators
        self.seed = random_state
        # for predictors weights
        self.predictor_weight = []

    def fit(self, X, y):
        ''' function to train AdaBoost '''
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y

        # initially all instances have the same weight
        instance_weights = np.full(shape=n_samples, fill_value=1/n_samples)

        for i in range(self.n_estimators):

            # Use predictor to make classification error
            clone_clf = DecisionTree(criterion='gini', max_depth=1)
            clone_clf.fit(X, y)
            predicted = clone_clf.predict(X)

            # getting misclassified instances
            misclassified_idx, acc = self.get_accuracy(y, predicted)
            # print(acc)

            # total error made by the predictor
            total_err = sum(instance_weights[misclassified_idx])
            # print("Error rate: ", len(misclassified_idx)/len(X))

            # weight of predictor based on its error
            EPS = 1e-10
            alpha = 0.5 * np.log((1.0 - total_err + EPS) / (total_err + EPS))
            self.predictor_weight.append((i, clone_clf, alpha))

            # updating instance weights
            instance_weights = self.update_instance_weights(
                misclassified_idx, instance_weights, alpha
            )

            # sampling data with replacement focusing on misclassified instances
            X, y = self.new_sample_set(X, y, instance_weights)

    def get_accuracy(self, true, predicted):
        ''' get prediction accuracy and which are misclassified '''
        assert len(true) == len(predicted)
        error_instance = np.equal(true.reshape(-1,), predicted)
        misclassified_idx = []

        for i, j in enumerate(error_instance):
            if j == 0:
                misclassified_idx.append(i)

        accuracy = np.sum(true.reshape(-1,) == predicted)
        return misclassified_idx, accuracy/len(true)

    def update_instance_weights(
        self, misclassified_instances,
        instance_weights, predictor_weight,
    ):
        ''' updating instance weights and normalize'''
        weights = instance_weights[:]
        for idx in range(len(instance_weights)):
            # w * exp(-alpha * pred * true)
            # avoid *0
            if idx in misclassified_instances:
                # pred * true = (-1, 1) => - * - = +
                weights[idx] *= np.exp(predictor_weight)
            else:
                # pred * true = (-1, -1) or (1, 1) => + * - = -
                weights[idx] *= np.exp(-predictor_weight)

        # Normalizing weights
        summed_weights = np.sum(weights)
        weights /= summed_weights
        return np.array(weights)

    def new_sample_set(self, X, y, instance_weights):
        ''' sampling data with replacement focusing on instances that were misclassified '''
        x = [i for i in range(len(X))]

        random.seed(self.seed)
        idx = random.choices(
            x, weights=instance_weights, k=len(instance_weights)
        )
        # n_idx = np.unique(idx)  # check replace

        X = self.X_train[idx]
        y = self.y_train[idx]
        return X, y

    def predict(self, X):
        ''' function to predict new dataset '''

        clf_predictions = np.array(
            [clf.predict(X) for idx, clf, weight in self.predictor_weight]
        )
        predictions = []

        for sample_predictions in clf_predictions.T:
            class_0 = 0
            class_1 = 0
            # label {0, 1}, zero * w = zero, so accumulate w
            for idx, score in enumerate(sample_predictions):
                if score <= 0:
                    class_0 += self.predictor_weight[idx][2]
                else:
                    class_1 += self.predictor_weight[idx][2]
            # Need to check your ground truth class is 0 and 1
            if class_0 > class_1:
                predictions.append(0)
            else:
                predictions.append(1)

        return np.array(predictions)

# ### Question 4.1
# Show the accuracy score of test data by `n_estimators=10` and `n_estimators=100`, respectively.


boost10 = AdaBoost(n_estimators=10)
boost10.fit(X_train, Y_train)
Y_pred = boost10.predict(X_test)
# print(Y_pred)
acc = accuracy_score(Y_test, Y_pred)
print("AdaBoost10: ", acc)


boost100 = AdaBoost(n_estimators=100)
boost100.fit(X_train, Y_train)
Y_pred = boost100.predict(X_test)
# print(Y_pred)
acc = accuracy_score(Y_test, Y_pred)
print("AdaBoost100: ", acc)
print("=============================================")

# ## Question 5
# implement the Random Forest algorithm by using the CART you just implemented from question 2.
# You should implement three arguments for the Random Forest.
# 1. **n_estimators**: The number of trees in the forest.
# 2. **max_features**: The number of random select features to consider when looking for the best split
# 3. **bootstrap**: Whether bootstrap samples are used when building tree
# - Note: Use majority votes to get the final prediction,
#         you may get slightly different results when re-building the random forest model
# Ref.: https://reurl.cc/vdg8ey


class RandomForest():
    """A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.

    Parameters:
    -----------
    n_estimators: int, default=10
        The number of weak classifiers that will be used.

    max_features : int or float, default=None
        The number of features to consider when looking for the best split.

    bootstrap: boolean, default="gini"
        Whether bootstrap samples are used when building trees.

    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    max_depth: int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure.
    """
    def __init__(
        self,
        n_estimators=10,
        max_features=None,
        bootstrap=True,
        criterion='gini',
        max_depth=None
    ):
        ''' constructor '''
        # initialize the parameters
        self.num_tree = n_estimators
        self.max_depth = max_depth
        self.n_feat = int(max_features)
        self.bootstrap = bootstrap
        self.criterion = criterion

    def fit(self, X, y):
        ''' function to train Random Forest '''
        self.tree = []
        for _ in range(self.num_tree):
            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.n_feat
            )

            # Choose data with the same number of duplicates or origin data
            if self.bootstrap:
                x_sample, y_sample = self.bootstrap_func(X, y)
            else:
                x_sample, y_sample = X, y

            tree.fit(x_sample, y_sample)
            self.tree.append(tree)

    def predict(self, X):
        ''' function to predict new dataset '''
        tree_predict = np.array([tree.predict(X) for tree in self.tree])
        tree_predict = np.swapaxes(tree_predict, 0, 1)
        # Voting
        y_pred = [self.most_commen_val(y) for y in tree_predict]
        return y_pred

    def bootstrap_func(self, X, y):
        ''' Whether bootstrap samples are used when building trees '''
        n_sample = X.shape[0]
        idexs = np.random.choice(n_sample,  n_sample, replace=True)
        # tmp = np.unique(idexs) checking whether puting back after taking
        return X[idexs], y[idexs]

    def most_commen_val(self, y):
        ''' Voting which class is often predited '''
        c = Counter(y)
        most = c.most_common(1)
        return most[0][0]


# ### Question 5.1
# Using `criterion=gini`, `max_depth=None`, `max_features=sqrt(n_features)`,
# showing the accuracy score of test data by `n_estimators=10` and `n_estimators=100`, respectively.

clf_10tree = RandomForest(
    n_estimators=10, max_features=np.sqrt(X_train.shape[1])
)
clf_10tree.fit(X_train, Y_train)
Y_pred = clf_10tree.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("clf_10tree: ", acc)

clf_100tree = RandomForest(
    n_estimators=100, max_features=np.sqrt(X_train.shape[1])
)
clf_100tree.fit(X_train, Y_train)
Y_pred = clf_100tree.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("clf_100tree: ", acc)
print("=============================================")

# ### Question 5.2
# Using `criterion=gini`, `max_depth=None`, `n_estimators=10`,
# showing the accuracy score of test data
# by `max_features=sqrt(n_features)` and `max_features=n_features`, respectively.

clf_random_features = RandomForest(
    n_estimators=10, max_features=np.sqrt(X_train.shape[1])
)
clf_random_features.fit(X_train, Y_train)
Y_pred = clf_random_features.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("clf_random_features: ", acc)

clf_all_features = RandomForest(n_estimators=10, max_features=X_train.shape[1])
clf_all_features.fit(X_train, Y_train)
Y_pred = clf_all_features.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("clf_all_features: ", acc)

print("\nTest begin===================================")
bootstrapF = RandomForest(
    n_estimators=10, max_features=np.sqrt(X_train.shape[1]), bootstrap=False
)
bootstrapF.fit(X_train, Y_train)
Y_pred = bootstrapF.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print("bootstrap_false: ", acc)
print("=====================================Test end\n")


# ### Question 6.
# Try you best to get highest test accuracy score by
# - Feature engineering
# - Hyperparameter tuning
# - Implement any other ensemble methods, such as gradient boosting.
#   Please note that you cannot call any package. Also, only ensemble method can be used.
#   Neural network method is not allowed to used.

q6 = RandomForest(
        n_estimators=100, max_features=np.sqrt(X_train.shape[1]), max_depth=4
    )
q6.fit(X_train, Y_train)
Y_pred = q6.predict(X_test)
print('Test-set accuarcy score: ', accuracy_score(Y_test, Y_pred))
