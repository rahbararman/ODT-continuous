import random
from matplotlib.pyplot import thetagrids
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
class Hypothesis:
    def __init__(self, value, is_active=True, decision_region=None):
        self.value = value
        self.is_active = is_active
        self.decision_region = decision_region


def binary(num, length):
    '''
        creates a name for each hypotheses
    '''
    return format(num, '#0{}b'.format(length + 2))[2:]



def compute_initial_h_probs(thetas, priors, hypothses):
    '''
    return a dictionary containing probabilities of each hypothesis (p(h)) with h's as keys
    parameters:
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        hypothses: ndarray containing all hypotheses (objects) 
    '''
    probs = {}
    for h in hypothses:
        p_h_y = 1
        for feature, value in enumerate(h.value):
            if int(float(value))==1:
                p_h_y = p_h_y * thetas[0][:,feature] 
            else:
                p_h_y = p_h_y * (1-thetas[0][:,feature])
        
        p_h = sum(priors*p_h_y)
        probs[h.value] = p_h
    return probs



def find_inconsistent_hypotheses(feature, hypotheses, feature_value):
    #checked
    '''
    returns a list of hypothesis inconsistent with feature observation (feature values)
    parameters:
        feature: the observed feature
        hypotheses: the list of hypotheses (objects)
        feature_value: observed feature value
    '''
    inconsistent_hypotheses = []
    for h in hypotheses:
        if str(h.value)[int(feature)] != str(int(feature_value)):
            inconsistent_hypotheses.append(h)
#     print([h.value for h in inconsistent_hypotheses])
    return inconsistent_hypotheses


def calculate_p_y_xA(thetas, priors, observations):
    #checked
    '''
    returns a ndarray containing p(y|x_A) for all ys
    parameters:
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        observations: a dictionary containing queried features as keys and  (thr_ind,value) as values.
    '''
    if (len(observations.items())==0):
        return priors
    #calculate p_xA
    p_xA_y = 1
    for feature, (thr_ind,value) in observations.items():
        if int(value)==1:
            p_xA_y = p_xA_y * thetas[thr_ind][:,int(feature)] 
        else:
            p_xA_y = p_xA_y * (1-thetas[thr_ind][:,int(feature)])
            
    
    p_xA = sum(priors*p_xA_y)
    p_y_xA = p_xA_y*priors/p_xA
    
#     print(p_y_xA)
    return p_y_xA


def calculate_p_feature_xA(feature, thetas, p_y_xA, feature_value):
    #checked
    '''
    parameters:
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        feature_value: (thr_ind, value) of the feature. 1 or 0
    '''
    if int(feature_value[1]) == 1:
        p_x_y = thetas[feature_value[0]][:, int(feature)]    
    else:
        p_x_y = 1 - thetas[feature_value[0]][:,int(feature)]
        
    p_feature_xA = sum(p_x_y*p_y_xA)
    
    return p_feature_xA

def calculate_expected_cut(feature,p_feature_xA, p_not_feature_xA, G, hypotheses):
    #checked
    #Need a proper way to find respective hypothesis
    '''
    parameters:
        hypotheses: the list of hypotheses (objects)
        p_feature_xA: P(x=1|x_A)
        p_not_feature_xA: P(x=0|x_A)
        G: the graph of hypotheses
    '''
    
    #step1: find the hypotheses inconsistent with feature
    hypotheses_feature = find_inconsistent_hypotheses(feature, hypotheses, 1)
    
    #step2: find the hypotheses inconsistent with not feature
    hypotheses_not_feature = find_inconsistent_hypotheses(feature, hypotheses, 0)
    
    #step3: Calculate the weights for each case
#     print(G.edges(data=True))
    edges_feature = G.edges(nbunch=[h.value for h in hypotheses_feature], data=True)
    sum_weights_feature = sum([w['weight'] for (u,v,w) in edges_feature])
    
    edges_not_feature = G.edges(nbunch=[h.value for h in hypotheses_not_feature], data=True)
    sum_weights_not_feature = sum([w['weight'] for (u,v,w) in edges_not_feature])
    
    #step4: Calculate the expectation
    expected_cut = p_feature_xA * sum_weights_feature + p_not_feature_xA *sum_weights_not_feature
    return expected_cut

def calculate_total_accuracy(thetas, thresholds, data, priors, theta_used_freq, metric='accuracy'):
    y_pred = []
    y_true = []
    
    for i in range(len(data)):
        # sampled_theta_ind = random.choice(range(len(thetas)))
        doc = data.iloc[i].to_dict()
        document_label = doc.pop('label', None)
        p_ob_y = 1
        for feature, value in doc.items():
            feature = int(float(feature))
            freqs_sum = np.sum(theta_used_freq, axis=0)
            thr_ind = np.argmax(freqs_sum[feature])
            if value > thresholds[thr_ind]:
                value = 1
            else:
                value = 0
            value = int(float(value))

            if value == 1:
                p_ob_y = p_ob_y * thetas[thr_ind][:,int(feature)]
            else:
                p_ob_y = p_ob_y * (1-thetas[thr_ind][:,int(feature)])
        y_pred.append(np.argmax(priors*p_ob_y))
        y_true.append(document_label)
    perf = 0.0
    if metric == 'accuracy':
        perf = accuracy_score(y_true, y_pred)
    if metric == 'fscore':
        perf = f1_score(y_true, y_pred, average='weighted')
    return perf


def estimate_priors_and_theta(dataset, rand_state):
    if dataset == '20newsgroup':
        vectorizer = TfidfVectorizer(stop_words='english',max_features=30)
        newsgroups = fetch_20newsgroups(subset='all')
        X = vectorizer.fit_transform(newsgroups.data).toarray()
        X = minmax_scale(X)
        X_train, X_test, y_train, y_test = train_test_split(X, newsgroups.target,test_size=0.3, random_state=rand_state)
        y_train = pd.DataFrame(y_train, columns=['label'])
        X_train = pd.DataFrame(X_train)
        y_test = pd.DataFrame(y_test, columns=['label'])
        X_test = pd.DataFrame(X_test)

    if dataset == 'diabetes':
        data = pd.read_csv('pima-indians-diabetes.csv', header=None)
        data.columns = list(range(len(data.columns)-1)) + ['label']
        np_from_data = data.to_numpy()
        np_from_data[:, :-1] = minmax_scale(np_from_data[:, :-1])
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.8, random_state=rand_state)
    
    
    
    data_csv = pd.concat([X_train,y_train], axis=1)
    test_csv = pd.concat([X_test,y_test], axis=1)
    
    num_features = data_csv.shape[1]-1
    num_classes = len(np.unique(test_csv['label'].to_numpy()))
    
    params = []
    for i in range(9):
        params.append(np.ones((num_classes, num_features, 2)))
    
    thetas = []
    for i in range(9):
        thetas.append(np.random.beta(params[i][:,:,0], params[i][:,:,1]))
    
    possible_ys = sorted(list(set(test_csv['label'].to_numpy())))
    priors = []
    for l in possible_ys:
        priors.append(1.0/len(possible_ys))

    theta_used_freq = np.ones((num_classes, num_features, 9))
        
    
    return params, thetas, np.array(priors), test_csv, data_csv, theta_used_freq

def calculate_expected_theta(thetas, theta_used_freq, label, feature):
    frequencies = theta_used_freq[label, feature,:]
    probs = frequencies/np.sum(frequencies)
    values = np.array([thetas[i][label, feature] for i in range(len(thetas))])
    return (values * probs).sum()


def create_dataset_for_efdt(dataset, rand_state):
    if dataset == '20newsgroup':
        vectorizer = TfidfVectorizer(stop_words='english',max_features=30)
        newsgroups = fetch_20newsgroups(subset='all')
        X = vectorizer.fit_transform(newsgroups.data).toarray()
        X = minmax_scale(X)
        X_train, X_test, y_train, y_test = train_test_split(X, newsgroups.target,test_size=0.3, random_state=rand_state)
        y_train = pd.DataFrame(y_train, columns=['label'])
        X_train = pd.DataFrame(X_train)
        y_test = pd.DataFrame(y_test, columns=['label'])
        X_test = pd.DataFrame(X_test)

    if dataset == 'diabetes':
        data = pd.read_csv('pima-indians-diabetes.csv', header=None)
        data.columns = list(range(len(data.columns)-1)) + ['label']
        np_from_data = data.to_numpy()
        np_from_data[:, :-1] = minmax_scale(np_from_data[:, :-1])
        data_new = pd.DataFrame(np_from_data, columns=data.columns)
        X_train, X_test, y_train, y_test = train_test_split(data_new.iloc[:,:-1], data_new['label'], test_size=0.8, random_state=rand_state)

    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()



def calculate_performance(y_true, y_pred, metric='accuracy'):
    if metric=="accuracy":
        return accuracy_score(y_true, y_pred)
    if metric=="fscore":
        return f1_score(y_true, y_pred, average='weighted')