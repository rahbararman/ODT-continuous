import random
import networkx as nx
import numpy as np
from discretize import find_best_threshold

from utils import Hypothesis, calculate_expected_cut, calculate_expected_theta, calculate_p_feature_xA, calculate_p_y_xA, compute_initial_h_probs, estimate_priors_and_theta, find_inconsistent_hypotheses
epsilon = 1.0
min_epsilon = 0.01
decay_rate = 1.0

def EC2(thresholds,h_probs, document, hypotheses, decision_regions, thetas, priors, observations, G=None, epsilon=0.0):

    '''
    Return the next feature to be queried and the current graph
    Parameters:
        G: the graph
        h_probs: a dictionary containing h_indices as keys and p(h|x_A) as values
        document: a dictionary containing feature names as keys and features as values
        hypotheses: ndarray of Hypothesis objects
        decision_regions: two dimentional list. first dimension is decision region and the second is the list of hyptheses in that region
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        priors: prior probabilities of decision regions (ys)
        observations: a dictionary containing queried features as keys and  (thr_ind,value) as values.
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
    best_thr_ind = 0
    rand_number = random.uniform(0,1)
    if rand_number < epsilon:
        return np.random.choice(list(document.keys())), G
    for feature in document.keys():
        p_y_xA = calculate_p_y_xA(thetas, priors, observations)
        thr_ind = find_best_threshold(thetas, observations, feature, priors, G, hypotheses, thresholds)
        p_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind,1))#P(x=1|x_A)
        p_not_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind,0))#P(x=0|x_A)
        expected_cut = calculate_expected_cut(feature, p_feature_xA, p_not_feature_xA, G, hypotheses)
        if (expected_cut > max_cut):
            max_cut = expected_cut
            best_feature = feature
            best_thr_ind = thr_ind
    return best_feature, thresholds[best_thr_ind], best_thr_ind, G

def IG(theta, priors, observations, document, epsilon=0.0):
    #step1: compute entropy(y|x_A)
    p_y_xA = calculate_p_y_xA(theta, priors, observations)
    temp = p_y_xA * np.log2(p_y_xA)
    entropy_y_xA = -sum(temp)
    
    
    #step2: for all features x compute IG(x)
    best_feature = None
    max_IG = float('-inf')
    rand_number = random.uniform(0,1)
    if rand_number <= epsilon:
        return np.random.choice(list(document.keys()))
    for feature in document.keys():
        #a. compute entropy(y|x_A,feature=1)
        new_observations = {feature:1}
        new_observations.update(observations)
        p_y_xA_feature = calculate_p_y_xA(theta, priors, new_observations)
        temp = p_y_xA_feature * np.log2(p_y_xA_feature)
        entropy_y_xA_feature = -sum(temp)
        #b. compute entropy(y|x_A,feature=0)
        new_observations = {feature:0}
        new_observations.update(observations)
        p_y_xA_not_feature = calculate_p_y_xA(theta, priors, new_observations)
        temp = p_y_xA_not_feature * np.log2(p_y_xA_not_feature)
        entropy_y_xA_not_feature = -sum(temp)
        #c. compute expected IG(feature)
        p_feature_xA =  calculate_p_feature_xA(feature, theta, p_y_xA, 1)#P(x=1|x_A)
        p_not_feature_xA =  calculate_p_feature_xA(feature, theta, p_y_xA, 0)#P(x=0|x_A)
        
        expected_IG = p_feature_xA*(entropy_y_xA-entropy_y_xA_feature)+p_not_feature_xA*(entropy_y_xA-entropy_y_xA_not_feature)
        if (expected_IG > max_IG):
            max_IG = expected_IG
            best_feature = feature
    return best_feature

def US(theta, priors, observations, document, h_probs, hypothses):
    #step1: compute entropy(H|x_A)
    p_h_xA = np.array(list(h_probs.values()))
    temp = p_h_xA * np.log2(p_h_xA)
    entropy_h_xA = -sum(temp)
    
    #step2: for all features x compute US(x)
    best_feature = None
    max_US = float('-inf')
    for feature in document.keys():
        #a. compute entropy(h|x_A, feature=1)
        new_observations = {feature:1}
        new_observations.update(observations)
        h_probs = {}
        p_y_xA = calculate_p_y_xA(theta, priors, new_observations)
        for h in hypothses:
            p_h_y = 1
            for feature_v, value in enumerate(h.value):
                if int(value)==1:
                    p_h_y = p_h_y * theta[:,feature_v] 
                else:
                    p_h_y = p_h_y * (1-theta[:,feature_v])

            p_xA_y = 1
            for feature_v, value in new_observations.items():
                if int(value)==1:
                    p_xA_y = p_xA_y * theta[:,int(feature_v)] 
                else:
                    p_xA_y = p_xA_y * (1-theta[:,int(feature_v)])

            p_h_xA_y = p_h_y/p_xA_y

            p_xA = sum(priors*p_xA_y)
            p_y_xA = p_xA_y*priors/p_xA

            p_h_xA = sum(p_y_xA*p_h_xA_y)
            h_probs[h.value] = p_h_xA
            
        p_h_xA_feature = np.array(list(h_probs.values()))
        temp = p_h_xA_feature * np.log2(p_h_xA_feature)
        entropy_h_xA_feature = -sum(temp)
        
        
        #a. compute entropy(h|x_A, feature=0)
        new_observations = {feature:0}
        new_observations.update(observations)
        h_probs = {}
        p_y_xA = calculate_p_y_xA(theta, priors, new_observations)
        for h in hypothses:
            p_h_y = 1
            for feature_v, value in enumerate(h.value):
                if int(value)==1:
                    p_h_y = p_h_y * theta[:,feature_v] 
                else:
                    p_h_y = p_h_y * (1-theta[:,feature_v])

            p_xA_y = 1
            for feature_v, value in new_observations.items():
                if int(value)==1:
                    p_xA_y = p_xA_y * theta[:,int(feature_v)] 
                else:
                    p_xA_y = p_xA_y * (1-theta[:,int(feature_v)])

            p_h_xA_y = p_h_y/p_xA_y

            p_xA = sum(priors*p_xA_y)
            p_y_xA = p_xA_y*priors/p_xA

            p_h_xA = sum(p_y_xA*p_h_xA_y)
            h_probs[h.value] = p_h_xA
        p_h_xA_not_feature = np.array(list(h_probs.values()))
        temp = p_h_xA_not_feature * np.log2(p_h_xA_not_feature)
        entropy_h_xA_not_feature = -sum(temp)
        
        
        
        #c. compute expected US(feature)
        p_y_xA = calculate_p_y_xA(theta, priors, observations)
        p_feature_xA =  calculate_p_feature_xA(feature, theta, p_y_xA, 1)#P(x=1|x_A)
        p_not_feature_xA =  calculate_p_feature_xA(feature, theta, p_y_xA, 0)#P(x=0|x_A)
        
        expected_US = p_feature_xA*(entropy_h_xA-entropy_h_xA_feature)+p_not_feature_xA*(entropy_h_xA-entropy_h_xA_not_feature)
        if expected_US > max_US:
            max_US = expected_US
            best_feature = feature
            
    return best_feature



def decision_tree_learning(thresholds,params, document, thetas, max_steps, priors, hypothses, decision_regions, criterion, theta_used_freq): 
    '''
    Receives a document and builds a decision tree with the EC2 algorithm.
    Parameters:
        criterion: the method to choose next feature to be queried
        document: the document to be classified. A dictionary containing feature names as keys and features as values
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        max_steps: the maximum number of features to be queried
        priors: prior probabilities of decision regions (ys)
        hypothses: ndarray of Hypothesis objects
        decision regions
    '''
    
    num_features = len(document.keys())
    num_labels = thetas[0].shape[0]
    h_probs = compute_initial_h_probs(thetas, priors, hypothses) #using the naive bayes assumption and summing over all class labels
    observations = {}
    G = None
    document_label = document.pop('label', None)
    for step in range(max_steps):
        if ('EC2' in criterion):
            if (criterion == "EC2_epsgreedy"):
                feature_to_be_queried, thr, thr_ind, G = EC2(thresholds, h_probs,document,hypothses,decision_regions, thetas, priors, observations, G, epsilon)
            else:
                feature_to_be_queried, thr, thr_ind, G = EC2(thresholds, h_probs,document,hypothses,decision_regions, thetas, priors, observations, G, 0.0)
            #query the next feature.
            feature_value = document[feature_to_be_queried]
            if feature_value > thr:
                feature_value = 1
            else:
                feature_value = 0
            feature_value = int(float(feature_value))
            observations[feature_to_be_queried] = (thr_ind,(int(float(feature_value))))

            
            #remove inconsistent hypotheses
            inconsistent_hypotheses = find_inconsistent_hypotheses(feature_to_be_queried, hypothses,feature_value)
            for inconsistenthypo in inconsistent_hypotheses:
                hypothses = [h for h in hypothses if h.value!=inconsistenthypo.value]
            
            #update p(h|x_A)
            h_probs = {}
            p_y_xA = calculate_p_y_xA(thetas, priors, observations)
            for h in hypothses:
                p_h_y = 1
                for feature, value in enumerate(h.value):
                    expected_theta_feature = np.array([calculate_expected_theta(thetas, theta_used_freq, y_i, feature) for y_i in range(num_labels)])
                    if int(value)==1:
                        p_h_y = p_h_y * expected_theta_feature
                    else:
                        p_h_y = p_h_y * (1-expected_theta_feature)

                p_xA_y = 1
                for feature, (thr_ind,value) in observations.items():
                    if int(value)==1:
                        p_xA_y = p_xA_y * thetas[thr_ind][:,int(feature)] 
                    else:
                        p_xA_y = p_xA_y * (1-thetas[thr_ind][:,int(feature)])

                p_h_xA_y = p_h_y/p_xA_y

                p_xA = sum(priors*p_xA_y)
                p_y_xA = p_xA_y*priors/p_xA

                p_h_xA = sum(p_y_xA*p_h_xA_y)
                h_probs[h.value] = p_h_xA
            


            #update the graph
            G = nx.Graph()
            for i in range(len(hypothses)):
                for j in range(i+1, len(hypothses)):
                    if hypothses[i].decision_region != hypothses[j].decision_region:
                        G.add_edge(hypothses[i].value, hypothses[j].value, weight=h_probs[hypothses[i].value]*h_probs[hypothses[j].value])
            
            if len(G.edges) == 0:
                break
        if ("IG" in criterion):
            if (criterion == "IG_epsgreedy"):
                feature_to_be_queried = IG(theta, priors, observations, document, epsilon)
            else:
                feature_to_be_queried = IG(theta, priors, observations, document, 0.0)
            feature_value = document[feature_to_be_queried]
            feature_value = int(float(feature_value))
            inconsistent_hypotheses = find_inconsistent_hypotheses(feature_to_be_queried, hypothses,feature_value)
            for inconsistenthypo in inconsistent_hypotheses:
                hypothses = [h for h in hypothses if h.value!=inconsistenthypo.value]
            one_region = True
            if len(hypothses) == 0:
                break
            re = hypothses[0].decision_region
            for hypo in hypothses:
                if hypo.decision_region != re:
                    one_region = False
                    break  
            observations[feature_to_be_queried] = int(float(feature_value))
            del document[feature_to_be_queried]
            if one_region:
                break
        if (criterion == 'US'):
            feature_to_be_queried = US(theta, priors, observations, document, h_probs, hypothses)
            feature_value = document[feature_to_be_queried]
            feature_value = int(float(feature_value))
            inconsistent_hypotheses = find_inconsistent_hypotheses(feature_to_be_queried, hypothses,feature_value)
            for inconsistenthypo in inconsistent_hypotheses:
                hypothses = [h for h in hypothses if h.value!=inconsistenthypo.value]
            one_region = True
            if len(hypothses) == 0:
                break
            re = hypothses[0].decision_region
            for hypo in hypothses:
                if hypo.decision_region != re:
                    one_region = False
                    break  
            observations[feature_to_be_queried] = int(float(feature_value))
            del document[feature_to_be_queried]
            if one_region:
                break
        
    
    
    #predict the label based on observations
    #y_hat = argmax_y p(y|observations)
    p_ob_y = 1
    for feature, (thr_ind,value) in observations.items():
        if int(value)==1:
            p_ob_y = p_ob_y * thetas[thr_ind][:,int(feature)] 
        else:
            p_ob_y = p_ob_y * (1-thetas[thr_ind][:,int(feature)])
    y_hat = np.argmax(priors*p_ob_y)  
    
    
    
    #observe the label
    y = int(document_label)
    
    for feature, (thr_ind,value) in observations.items():
        theta_used_freq[y, feature, thr_ind] = theta_used_freq[y, feature, thr_ind] + 1
        if int(value)==1:
            params[thr_ind][int(y), int(feature), 0] += 1
            
        else:
            params[thr_ind][int(y), int(feature), 1] += 1
        
        
    
    
    
    return observations, y, y_hat

def sample_hypotheses(N, thetas, priors, random_state, total_samples, theta_used_freq):
    #sampling hypotheses and generating decision regions
    #step1: sample y1,y2,...,yN from priors
    np.random.seed(random_state)
    num_features = thetas[0].shape[1]
    num_labels = thetas[0].shape[0]
    sampled_ys = []
    for n in range(N):
        y_n = np.random.choice(a = len(priors), p=priors)
        sampled_ys.append(y_n)
    #step2: sample h1,h2,...,hN from p(x|y)

    decision_regions = {}
    hypothses = []
    observed_hypothses = []
    for y_n in sampled_ys:
        #sample h
        while (True):
            sampled_h = ''
            for f in range(num_features):
                expected_theta_ij = calculate_expected_theta(thetas, theta_used_freq, y_n, f)
                generated_feature = np.random.choice(a=[0,1], p=[1-expected_theta_ij,expected_theta_ij])
                sampled_h = sampled_h + str(generated_feature)
            #determine region for sampled hypothesis
            #region of h_i = argmax_j p(y_j|h_i) based on theta
            #1.compute p(y|sampled h) for all y
            p_h_y = 1
            for feature, value in enumerate(sampled_h):
                value = int(value)
                expected_theta_feature = np.array([calculate_expected_theta(thetas, theta_used_freq, y_i, feature) for y_i in range(num_labels)])
                if value==1:
                    p_h_y = p_h_y * expected_theta_feature
                else:
                    p_h_y = p_h_y * (1-expected_theta_feature)

            region = np.argmax(priors*p_h_y)
            if not (sampled_h in observed_hypothses):
                observed_hypothses.append(sampled_h)
                new_h = Hypothesis(sampled_h)
                hypothses.append(new_h)
                new_h.decision_region = region

                if not (region in decision_regions.keys()):
                    decision_regions[region] = set()
                    decision_regions[region].add(sampled_h)
                else:
                    decision_regions[region].add(sampled_h)
                break
    return hypothses, decision_regions





