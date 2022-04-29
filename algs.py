import random
import networkx as nx
import numpy as np

from utils import calculate_expected_cut, calculate_p_feature_xA, calculate_p_y_xA

def EC2(h_probs, document, hypotheses, decision_regions, theta, priors, observations, G=None, epsilon=0.0):
    '''
    Return the next feature to be queried and the current graph
    Parameters:
        G: the graph
        h_probs: a dictionary containing h_indices as keys and p(h|x_A) as values
        document: a dictionary containing feature names as keys and features as values
        hypotheses: ndarray of Hypothesis objects
        decision_regions: two dimentional list. first dimension is decision region and the second is the list of hyptheses in that region
        theta: the condictional probabilites. m*n ndarray where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        observations: a dictionary containing queried features and respective values 
    note: hypothesis names are the respective decimal number of the binary realization
    '''
    #building the graph if it is none
    if G is None:
        G = nx.Graph()
        for i in range(len(hypotheses)):
            for j in range(i+1, len(hypotheses)):
                if hypotheses[i].decision_region != hypotheses[j].decision_region:
                    G.add_edge(hypotheses[i].value, hypotheses[j].value, weight=h_probs[hypotheses[i].value]*h_probs[hypotheses[j].value])
                
    #select the feature
    best_feature = None
    max_cut = float('-inf')
    rand_number = random.uniform(0,1)
    if rand_number < epsilon:
        return np.random.choice(list(document.keys())), G
    for feature in document.keys():
        p_y_xA = calculate_p_y_xA(theta, priors, observations)
        p_feature_xA =  calculate_p_feature_xA(feature, theta, p_y_xA, 1)#P(x=1|x_A)
        p_not_feature_xA =  calculate_p_feature_xA(feature, theta, p_y_xA, 0)#P(x=0|x_A)
        expected_cut = calculate_expected_cut(feature, p_feature_xA, p_not_feature_xA, G, hypotheses)
        if (expected_cut > max_cut):
            max_cut = expected_cut
            best_feature = feature
    return best_feature, G



