import numpy as np
import sys

from scipy.io import arff
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing

from sklearn import linear_model
from sklearn.lda import LDA
import collections


SIZE = 20


# *********************** Main program ************************ #
if ( len(sys.argv) < 2 ):
  print("python cross_validation.py train [mi] [ma] [grid]")
  sys.exit()


# Open and access the file
tr_f = sys.argv[1]


mi = 4
ma = 5

if ( len(sys.argv) > 2 ):
    mi = int(sys.argv[2])
    ma = int(sys.argv[3])


opcao = 0
if ( len(sys.argv) > 3 ):
    opcao = 1


print mi, ma

print 'Starting read from Train file...'
#temp = np.loadtxt(tr_f, dtype=float)
temp = np.genfromtxt(tr_f, dtype='str')
print temp.shape
f_train, f_train_patients, f_train_labels = np.hsplit(temp, np.array([mi, ma]))
#f_train, f_train_labels, oto = np.hsplit(temp, np.array([mi, ma]))
print 'train.shape = {0}'.format(f_train.shape)
print 'train_labels.shape = {0}'.format(f_train_labels.shape)

f_train = np.array( [[float(y) for y in x] for x in f_train] )

u, ind = np.unique(f_train_patients, return_index=True)

#for i in ind:
#    print f_train_labels[i]


ini,l = f_train.shape 
prop = int ((40 * ini)/100)


dt = np.array([])
gnb = np.array([])
mnb = np.array([])
bnb = np.array([])
knn = np.array([])
sv = np.array([])
per = np.array([])
ldan = np.array([])
lr_ovr = np.array([])
lr_mul = np.array([])

knn_h = {}
k_range = [5, 9, 11, 13]
for i in k_range:
    knn_h[i] = 0.0



def GridSearch(X_train, y_train):

        # define range dos parametros
        C_range = 2. ** np.arange(-5,15,2)
        gamma_range = 2. ** np.arange(3,-15,-2)
        k = [ 'rbf']
        #k = ['linear', 'rbf']
        param_grid = dict(gamma=gamma_range, C=C_range, kernel=k)

        # instancia o classificador, gerando probabilidades
        srv = svm.SVC(probability=True)

        # faz a busca
        grid = GridSearchCV(srv, param_grid, n_jobs=-1, verbose=True)
        print "grid.fit"
        grid.fit (X_train, (y_train.ravel()) )

        # recupera o melhor modelo
        model = grid.best_estimator_

        # imprime os parametros desse modelo
        print grid.best_params_
        print model
        return model
        


count = (-1)

var = ((i[0], 0) for i in f_train_patients)
var = dict(var)

for i in range(0, SIZE):
    
    
    count += 1

    train = f_train
    train_labels = f_train_labels
    test = np.array([])
    test_labels = np.array([])
    train_patients = f_train_patients
    
    # Aleatorio
    #   Validacao cruzada 10 vezes - prop: 60/40
    #for j in range(0, prop):
    #    a = train.shape[0] - 1
    #    rand = random.randint(0, a)
    
    '''
    # Retira um paciente de cada classe
    each = { '1': 0, '2': 0, '3': 0, '4': 0, '5': 0 }
    j = 0; still = 1
    old = f_train_patients[j][0]
    len_t, n = train_labels.shape
    while ( still and j < len_t ) :  

        len_t, n = train_labels.shape
        new = f_train_patients[j][0]
        if ( new != old ):
            each[train_labels[j][0]] += 1

        if ( each[train_labels[j][0]] == count ):
            
            rand = j
            test = np.append(test, train[rand])
            test_labels = np.append(test_labels, train_labels[rand])
        
            train = np.delete(train, rand, 0)
            train_labels = np.delete(train_labels, rand, 0)

        kc = 0
        for k, v in each.iteritems(): 
            if ( v > count ):
                kc += 1
        if ( kc == 5 ):
            still = 0

        len_t, n = train_labels.shape
        old = new
        j += 1
    '''

    # Retira um paciente contra todos os outros (19)
    j = 0
    foi = 0; mudou = 0
    old = train_patients[j][0]
    len_t, n = train_labels.shape
    while ( j < len_t ) :  

        new = train_patients[j][0]
        if ( new != old ):
            if ( mudou and (not foi) ):
                foi = 1
                var[train_patients[j-2][0]] = 1

        if ( var[train_patients[j][0]] == 0 and (not foi) ):
            mudou += 1
            
            rand = j
            test = np.append(test, train[rand])
            test_labels = np.append(test_labels, train_labels[rand])
        
            train = np.delete(train, rand, 0)
            train_labels = np.delete(train_labels, rand, 0)
            train_patients = np.delete(train_patients, rand, 0)
        
        len_t, n = train_labels.shape
        old = new
        j += 1


    '''
    train, test, train_labels, test_labels = train_test_split(f_train, f_train_labels, test_size=0.40, random_state=42)
    '''
    test = np.reshape(test, ((ini-train.shape[0]), l))
    print train.shape, test.shape 
    
    tr_labels = train_labels.ravel()
    te_labels = test_labels.ravel()


    # ********************* Funcoes ********************** #
    
    # Decision Tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train, train_labels)

    print ''
    print '******************************************************'
    print 'Decision Tree:'

    pred = np.array([])
    pred = clf.predict(test)
    
    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix(pred, test_labels)
    dt = np.append(dt, acc)

    # Gaussian Naive Bayes
    print ''
    print '******************************************************'
    print 'Gaussian Naive Bayes:'

    train_labels = np.ravel(train_labels)
    test_labels = np.ravel(test_labels)
    pred = np.array([])
    clf = GaussianNB()
    pred = clf.fit(train, train_labels).predict(test)
    
    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    gnb = np.append(gnb, acc)

    '''
    # Multinomial Naive Bayes
    print ''
    print '******************************************************'
    print 'Multinomial Naive Bayes:'

    pred = np.array([])
    clf = MultinomialNB()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    mnb = np.append(mnb, acc)


    # Bernoulli Naive Bayes
    print ''
    print '******************************************************'
    print 'Bernoulli Naive Bayes:'

    pred = np.array([])
    clf = BernoulliNB()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    bnb = np.append(bnb, acc)
    '''
    
    # KNN
    print ''
    print '******************************************************'
    print 'KNN:'


    for k in k_range:
        
        print '---------------------'
        print 'KNN == ', k, ':'
        
        pred = np.array([])
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train, train_labels) 
        pred = np.append(pred, (neigh.predict( test )) )

        acc = accuracy_score(pred, test_labels)
        print ''
        print acc
        print confusion_matrix( test_labels, pred )
        knn = np.append(knn, acc)
        knn_h[k] += acc
    
    
    # SVM
    print ''
    print '******************************************************'
    print 'SVM:'

    
    # GridSearch retorna o melhor modelo encontrado na busca
    if (opcao):
        best = GridSearch(train, train_labels)
    else:
        best = svm.SVC(C=2.0, gamma=0.0001220703125, kernel='rbf', probability=True)
        #{'kernel': 'rbf', 'C': 128.0, 'gamma': 0.0001220703125}
        #{'kernel': 'rbf', 'C': 2.0, 'gamma': 0.0001220703125}
        #{'kernel': 'rbf', 'C': 8.0, 'gamma': 0.0001220703125}

        #{'kernel': 'rbf', 'C': 2.0, 'gamma': 0.0001220703125}
    
    best.fit(train, tr_labels)
    acc = best.score(test, test_labels)
    
    print ''
    print acc
    #print confusion_matrix( test_labels, pred )
    sv = np.append(sv, acc)


    # Perceptron
    print ''
    print '******************************************************'
    print 'Perceptron:'

    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
    pred = np.array([])
    clf = linear_model.Perceptron()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    per = np.append(per, acc)


    # LDA
    print ''
    print '******************************************************'
    print 'LDA:'

    # http://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html
    pred = np.array([])
    clf = LDA()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    ldan = np.append(ldan, acc)


    # Logistic Regression - ovr
    print ''
    print '******************************************************'
    print 'Logistic Regression - ovr:'

    pred = np.array([])
    clf = linear_model.LogisticRegression()
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    lr_ovr = np.append(lr_ovr, acc)


    # Logistic Regression - multinomial
    print ''
    print '******************************************************'
    print 'Logistic Regression - mul:'

    pred = np.array([])
    clf = linear_model.LogisticRegression( solver='newton-cg', multi_class='multinomial' )
    clf.fit(train, train_labels)
    pred = clf.predict(test)

    acc = accuracy_score(pred, test_labels)
    print ''
    print acc
    print confusion_matrix( test_labels, pred )
    lr_mul = np.append(lr_mul, acc)
    
    print '######################################################'


print '######################################################'

print 'DT: '
print dt
print np.mean(dt)

print 'gnb: '
print gnb
print np.mean(gnb)
'''
print 'mnb: '
print mnb
print np.mean(mnb)

print 'bnb: '
print bnb
print np.mean(bnb)
'''
print 'knn: '
print knn
print np.mean(knn)
knn_h_ordered = collections.OrderedDict( knn_h )
for key in ( knn_h_ordered ):
    print key, ' = ', (knn_h_ordered[key] / SIZE)


print 'svm: '
print sv
print np.mean(sv)

'''
print 'per: '
print per
print np.mean(per)
'''
print 'lda: '
print ldan
print np.mean(ldan)

print 'lr_ovr: '
print lr_ovr
print np.mean(lr_ovr)

print 'lr_mul: '
print lr_mul
print np.mean(lr_mul)

