from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def DecisionTree(train_pattern,train_label,test_pattern):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_pattern, train_label)

    result = clf.predict(test_pattern)
    return result


def KNN(train_pattern,train_label,test_pattern):
    clf = KNeighborsClassifier()
    clf.fit(train_pattern,train_label)
    result = clf.predict(test_pattern)
    return result


def SVM(train_pattern,train_label,test_pattern):
    clf = SVC(kernel = 'rbf',probability = True, gamma = 'scale')
    clf.fit(train_pattern,train_label)
    result = clf.predict(test_pattern)
    return result


def NB(train_pattern, train_label,test_pattern):
    clf = MultinomialNB(alpha = 0.1)
    clf.fit(train_pattern,train_label)
    result = clf.predict(test_pattern)
    return result


def RF(train_pattern, train_label, test_pattern):
    clf =  RandomForestClassifier(n_estimators=8)
    clf.fit(train_pattern, train_label)
    result = clf.predict(test_pattern)
    return result
