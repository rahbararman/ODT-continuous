from ast import Return
from utils import calculate_expected_cut, calculate_expected_theta, calculate_p_feature_xA, calculate_p_y_xA
import numpy as np

def calculate_threshold_sampling_distribution(S_tresholds, feature, eta=0.01, need_log=False):
    if not need_log:
        pi = np.exp(eta*S_tresholds[feature,:])/np.sum(np.exp(eta*S_tresholds[feature,:]))
    else: 
        pi = np.exp(eta*np.log(S_tresholds[feature,:]+0.0001))/np.sum(np.exp(eta*np.log(S_tresholds[feature,:]+0.0001)))
    if np.isnan(pi).any():
        print('this is S_thresholds')
        print(S_tresholds[feature,:])
        print('this is pi')
        print(pi)
    return pi


def find_best_threshold_EC2(thetas, observations, feature, priors, G, hypotheses, thresholds, S_thresholds,exhaustive=True):
    """
    returns the index of threshold to binarize the feature with respect to EC2 (and prob. of that threshold). 
    parameters:
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        observations: a dictionary containing queried features as keys and  (thr_ind,value) as values.
        feature: the feature to binarize
        priors: the prior probs for decision regions.
        G: the graph
        hypotheses: the remaining hypotheses
    """
    if exhaustive:
        best_thr_ind = None
        max_cut = float('-inf')

        for thr_ind in range(len(thresholds)):
            #find the expected gain for thresholds[i]
            #Need to find p(feature_i|x_A) for values 0 and 1 
            p_y_xA = calculate_p_y_xA(thetas, priors, observations)
            p_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind,1))#P(x=1|x_A)
            p_not_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind,0))#P(x=0|x_A)
            expected_cut = calculate_expected_cut(feature, p_feature_xA, p_not_feature_xA, G, hypotheses)
            if (expected_cut > max_cut):
                max_cut = expected_cut
                best_thr_ind = thr_ind
        return best_thr_ind, None

    else:
        threshold_sampling_distribution = calculate_threshold_sampling_distribution(S_thresholds,feature, need_log=True)
        best_thr_ind = np.random.choice(len(thresholds), p=threshold_sampling_distribution)
        return best_thr_ind, threshold_sampling_distribution[best_thr_ind]

def find_best_threshold_IG(thetas, observations, feature, priors, thresholds,  p_y_xA, entropy_y_xA, S_thresholds, exhaustive=True):
    """
    returns the index of threshold to binarize the feature with respect to IG (and prob. of that threshold). 
    parameters:
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        observations: a dictionary containing queried features as keys and  (thr_ind,value) as values.
        feature: the feature to binarize
        priors: the prior probs for decision regions.
        hypotheses: the remaining hypotheses
        thresholds: the list of thresholds to use
    """
    if exhaustive:
        best_thr_ind = None
        max_IG = float('-inf')

        for thr_ind in range(len(thresholds)):
            #find the best expected gain
            new_observations = {feature:(thr_ind,1)}
            new_observations.update(observations)
            p_y_xA_feature = calculate_p_y_xA(thetas, priors, new_observations)
            temp = p_y_xA_feature * np.log2(p_y_xA_feature)
            entropy_y_xA_feature = -sum(temp)

            new_observations = {feature:(thr_ind,0)}
            new_observations.update(observations)
            p_y_xA_not_feature = calculate_p_y_xA(thetas, priors, new_observations)
            temp = p_y_xA_not_feature * np.log2(p_y_xA_not_feature)
            entropy_y_xA_not_feature = -sum(temp)
            #c. compute expected IG(feature)
            p_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind, 1))#P(x=1|x_A)
            p_not_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind, 0))#P(x=0|x_A)
            
            expected_IG = p_feature_xA*(entropy_y_xA-entropy_y_xA_feature)+p_not_feature_xA*(entropy_y_xA-entropy_y_xA_not_feature)
            if (expected_IG > max_IG):
                max_IG = expected_IG
                best_thr_ind = thr_ind

        return best_thr_ind, None
    else:
        threshold_sampling_distribution = calculate_threshold_sampling_distribution(S_thresholds,feature)
        best_thr_ind = np.random.choice(len(thresholds), p=threshold_sampling_distribution)
        return best_thr_ind, threshold_sampling_distribution[best_thr_ind]


def find_best_threshold_US(thetas, observations, feature, priors, thresholds, hypothses, theta_used_freq, entropy_h_xA, S_thresholds, exhaustive=True):
    
    if exhaustive:
        best_thr_ind = None
        max_US = float('-inf')

        for thr_ind in range(len(thresholds)):
            new_observations = {feature:(thr_ind,1)}
            new_observations.update(observations)
            h_probs = {}
            p_y_xA = calculate_p_y_xA(thetas, priors, new_observations)
            for h in hypothses:
                p_h_y = 1
                for feature_v, value in enumerate(h.value):
                    expected_theta_feature_v = np.array([calculate_expected_theta(thetas, theta_used_freq, y_i, feature_v) for y_i in range(len(priors))])
                    if int(value)==1:
                        p_h_y = p_h_y * expected_theta_feature_v
                    else:
                        p_h_y = p_h_y * (1-expected_theta_feature_v)

                p_xA_y = 1
                for feature_v, (thr_ind,value) in new_observations.items():
                    if int(value)==1:
                        p_xA_y = p_xA_y * thetas[thr_ind][:,int(feature_v)] 
                    else:
                        p_xA_y = p_xA_y * (1-thetas[thr_ind][:,int(feature_v)])

                p_h_xA_y = p_h_y/p_xA_y

                p_xA = sum(priors*p_xA_y)
                p_y_xA = p_xA_y*priors/p_xA

                p_h_xA = sum(p_y_xA*p_h_xA_y)
                h_probs[h.value] = p_h_xA
                
            p_h_xA_feature = np.array(list(h_probs.values()))
            temp = p_h_xA_feature * np.log2(p_h_xA_feature)
            entropy_h_xA_feature = -sum(temp)
            
            
            #a. compute entropy(h|x_A, feature=0)
            new_observations = {feature:(thr_ind,0)}
            new_observations.update(observations)
            h_probs = {}
            p_y_xA = calculate_p_y_xA(thetas, priors, new_observations)
            for h in hypothses:
                p_h_y = 1
                for feature_v, value in enumerate(h.value):
                    expected_theta_feature_v = np.array([calculate_expected_theta(thetas, theta_used_freq, y_i, feature_v) for y_i in range(len(priors))])
                    if int(value)==1:
                        p_h_y = p_h_y * expected_theta_feature_v 
                    else:
                        p_h_y = p_h_y * (1-expected_theta_feature_v)

                p_xA_y = 1
                for feature_v, (thr_ind,value) in new_observations.items():
                    if int(value)==1:
                        p_xA_y = p_xA_y * thetas[thr_ind][:,int(feature_v)] 
                    else:
                        p_xA_y = p_xA_y * (1-thetas[thr_ind][:,int(feature_v)])

                p_h_xA_y = p_h_y/p_xA_y

                p_xA = sum(priors*p_xA_y)
                p_y_xA = p_xA_y*priors/p_xA

                p_h_xA = sum(p_y_xA*p_h_xA_y)
                h_probs[h.value] = p_h_xA
            p_h_xA_not_feature = np.array(list(h_probs.values()))
            temp = p_h_xA_not_feature * np.log2(p_h_xA_not_feature)
            entropy_h_xA_not_feature = -sum(temp)
            
            
            
            #c. compute expected US(feature)
            p_y_xA = calculate_p_y_xA(thetas, priors, observations)
            p_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind,1))#P(x=1|x_A)
            p_not_feature_xA =  calculate_p_feature_xA(feature, thetas, p_y_xA, (thr_ind,0))#P(x=0|x_A)
            
            expected_US = p_feature_xA*(entropy_h_xA-entropy_h_xA_feature)+p_not_feature_xA*(entropy_h_xA-entropy_h_xA_not_feature)

            if (expected_US > max_US):
                max_US = expected_US
                best_thr_ind = thr_ind
    
        return best_thr_ind, None
    else:
        threshold_sampling_distribution = calculate_threshold_sampling_distribution(S_thresholds,feature)
        best_thr_ind = np.random.choice(len(thresholds), p=threshold_sampling_distribution)
        return best_thr_ind, threshold_sampling_distribution[best_thr_ind]








        

    




