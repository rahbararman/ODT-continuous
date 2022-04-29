import numpy as np
import networkx as nx
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



def compute_initial_h_probs(theta, priors, hypothses):
    '''
    return a dictionary containing probabilities of each hypothesis (p(h)) with h's as keys
    parameters:
        theta: the condictional probabilites. m*n ndarray where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        hypothses: ndarray containing all hypotheses (objects) 
    '''
    probs = {}
    for h in hypothses:
        p_h_y = 1
        for feature, value in enumerate(h.value):
            if int(float(value))==1:
                p_h_y = p_h_y * theta[:,feature] 
            else:
                p_h_y = p_h_y * (1-theta[:,feature])
        
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


def calculate_p_y_xA(theta, priors, observations):
    #checked
    '''
    returns a ndarray containing p(y|x_A) for all ys
    parameters:
        theta: the condictional probabilites. m*n ndarray where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        observations: a dictionary containing queried features and respective values 
    '''
    if (len(observations.items())==0):
        return priors
    #calculate p_xA
    p_xA_y = 1
    for feature, value in observations.items():
        if int(value)==1:
            p_xA_y = p_xA_y * theta[:,int(feature)] 
        else:
            p_xA_y = p_xA_y * (1-theta[:,int(feature)])
            
    
    p_xA = sum(priors*p_xA_y)
    p_y_xA = p_xA_y*priors/p_xA
    
#     print(p_y_xA)
    return p_y_xA


def calculate_p_feature_xA(feature, theta, p_y_xA, feature_value):
    #checked
    '''
    parameters:
        theta: the condictional probabilites. m*n ndarray where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        feature_value: value of the feature. 1 or 0
    '''
    if int(feature_value) == 1:
        p_x_y = theta[:, int(feature)]    
    else:
        p_x_y = 1 - theta[:,int(feature)]
        
    p_feature_xA = sum(p_x_y*p_y_xA)
    
#     print(p_feature_xA)
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
    
#     print(sum_weights_feature)
#     print(p_feature_xA)
#     print(sum_weights_not_feature)
#     print(p_not_feature_xA)
    
    #step4: Calculate the expectation
    expected_cut = p_feature_xA * sum_weights_feature + p_not_feature_xA *sum_weights_not_feature
#     print(expected_cut)
    return expected_cut


