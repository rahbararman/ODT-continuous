from utils import calculate_expected_cut, calculate_p_feature_xA, calculate_p_y_xA


def find_best_threshold(thetas, observations, feature, priors, G, hypotheses, thresholds):
    """
    returns the index of threshold to binarize the feature. 
    parameters:
        thetas: the condictional probabilites. a list of m*n ndarrays where m is the number of decision regions and n is the number of features
        observations: a dictionary containing queried features as keys and  (thr_ind,value) as values.
        feature: the feature to binarize
        priors: the prior probs for decision regions.
        G: the graph
        hypotheses: the remaining hypotheses
    """
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
        
    return best_thr_ind




        

    




