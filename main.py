import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
import numpy as np
import pandas as pd
import time
from graphviz import Digraph
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("secondary_data.csv", delimiter=';')
df.head()

data_encoded = pd.get_dummies(df,dtype = int)
data_encoded.head()

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=None, max_leaf_nodes=None, splitting_threshold=None, split_function=None, min_samples_split=2, feature_names=None):
        
        self.root = None
        
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.splitting_threshold = splitting_threshold
        self.split_function = split_function
        self.min_samples_split = min_samples_split
        self.feature_names = feature_names
        self.leaf_count = 0
        self.depth = 0

        
        self.criterion_func = {
            'scaled_entropy': self._scaled_entropy,
            'gini': self._gini_impurity,
            'squared': self._squared_impurity,
        }.get(self.split_function)

    def get_params(self, deep=True):
        return {
            'max_depth': self.max_depth,
            'max_leaf_nodes': self.max_leaf_nodes,
            'splitting_threshold': self.splitting_threshold,
            'split_function': self.split_function,
            'min_samples_split': self.min_samples_split,
            'feature_names': self.feature_names
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)



    #building of tree(recursive)
    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape

        current_threshold = self.criterion_func(y)

        if depth > self.depth:
            self.depth = depth

        if (self.max_depth is not None and depth >= self.max_depth) \
                or (self.max_leaf_nodes is not None and self.leaf_count >= self.max_leaf_nodes) \
                or (self.splitting_threshold is not None and current_threshold < self.splitting_threshold) \
                or (num_samples < self.min_samples_split) \
                or (np.unique(y).size == 1):
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        feat_idxs = np.random.choice(num_features, num_features, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        self.leaf_count += 1
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return TreeNode(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh
    

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gain(self, y, X_column, split_thresh):

        parent_criterion = self.criterion_func(y)

        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        y_left, y_right = y[left_idxs], y[right_idxs]
        child_criterion = self._weighted_criterion(y_left, y_right, self.criterion_func)
        gain = parent_criterion - child_criterion
        return gain

    def _weighted_criterion(self, y_left, y_right, criterion_func):
        n = len(y_left) + len(y_right)
        p_left = len(y_left) / n
        p_right = len(y_right) / n
        return p_left * criterion_func(y_left) + p_right * criterion_func(y_right)

    # Raw SCALED ENTRPY FORMULA
    def _scaled_entropy(self, y):
        hist = np.bincount(y)
        probs = hist / len(y)
        scaled_ent = -np.sum([(p / 2) * np.log2(p) for p in probs if p > 0])
        return scaled_ent


    #Raw GINI IMPURITY FORMULA
    #For binary classification it should be 2p(1-p) but results are a little awkward
    def _gini_impurity(self, y):
        hist = np.bincount(y)
        probs = hist / len(y)
        gini = 1.0 - np.sum(probs ** 2)  # gini = np.sum(probs * (1 - probs))
        return gini

    # for binary classification    sqrt(p*(1-p))
    def _squared_impurity(self, y):
        hist = np.bincount(y)
        probs = hist / len(y)
        epsilon = 1e-10  # penalty to avoid multiplying by zero
        sqr = np.sum(np.sqrt((probs + epsilon) * (1 - probs + epsilon)))
        return sqr

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def visualize_tree(self, dot=None):
        if dot is None:
            dot = Digraph()

        # Recursive function to add nodes and edges
        def add_nodes_edges(dot, node):
            if node is None:
                return
            if node.is_leaf():
                dot.node(str(id(node)), f"Class {node.value}", shape='ellipse', style='filled', fillcolor='#fc844c')
            else:
                if self.feature_names is not None:
                    feature_name = self.feature_names[node.feature]
                else:
                    feature_name = f"Feature {node.feature}"
                dot.node(str(id(node)), f"{feature_name} <= {node.threshold:.2f}", shape='doubleoctagon', style='filled', fillcolor='lightblue')
                if node.left is not None:
                    add_nodes_edges(dot, node.left)
                    dot.edge(str(id(node)), str(id(node.left)))
                if node.right is not None:
                    add_nodes_edges(dot, node.right)
                    dot.edge(str(id(node)), str(id(node.right)))

        add_nodes_edges(dot, self.root)
        return dot


def zero_one_loss(y_true, y_pred):
    return np.mean(y_pred != y_true)


if __name__ == '__main__':

    data = pd.read_csv('secondary_data.csv', delimiter=';')



    # Convert categorical features to one-hot encoding
    data_encoded = pd.get_dummies(data)
    data_encoded = data_encoded.astype(int)
  
    print(data_encoded.head())


    # Separate features and target
    X = data_encoded.drop(['class_p', 'class_e'], axis=1)
    y = data_encoded['class_p']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



tree = DecisionTree(
        max_depth=8,
        max_leaf_nodes=None,
        splitting_threshold=0.1,
        split_function="scaled_entropy",
        min_samples_split=2,  
        feature_names=X_train.columns
    )

start_time = time.time()
tree.fit(X_train.values, y_train.values)
end_time = time.time()
execution_time = end_time - start_time

print(f"\nExecution time: {execution_time} seconds")

# Predict on the testing data
y_test_pred = tree.predict(X_test.values)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test.values, y_test_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\n\n")

# Visualize the tree
dot = Digraph()
dot = tree.visualize_tree(dot)
dot.render('png/mushroom_tree', format='png', view=True)




# Predicted training data
y_pred = tree.predict(X_train.values)

# Evaluate the model on the training data
accuracy = accuracy_score(y_train, y_pred)
print(f"\nTrain accuracy: {accuracy:.6f}")


# Evaluate the model on the testing data
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy: {accuracy:.6f}")

# Zero-one loss
train_error = zero_one_loss(y_test.values, y_test_pred)
print(f"0 - 1 Loss: {train_error:.6f}")




