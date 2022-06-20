from dataclasses import replace
import numpy as np
import random


def truncate_weights(weights, num_features):
    weights = np.copy(weights)
    if np.count_nonzero(weights) > num_features:
        weights[np.flip(np.argsort(weights))[num_features:]] = 0
    
    return weights

def ofs_partial(R, num_features, eps, stepsize, X, Y):
    '''
        R: maximum L2 norm
        num_features: number of selected features (B in the paper)
        eps: exploration exploitation tradeoff (epsilon in the paper)
        stepsize: step size
        X: dataset features num_samples*total_num_features
        Y: dataset labels
    '''
    #initialization
    totol_num_features = X.shape[1]
    selected_features = np.random.choice(totol_num_features, replace=False, size=num_features)
    weights = np.zeros(totol_num_features)
    weights[selected_features] += 0.01
    y_pred = []
    num_mistakes = 0
    print('Number of samples: ')
    print(X.shape[0])
    print("************************************")
    count=0
    for x, y in zip(X, Y):
        if (count%100 == 0):
            print(count)
            print(len(selected_features))
        count+=1
        rand_number = random.uniform(0,1)
        if rand_number < eps:
            selected_features = np.random.choice(totol_num_features, replace=False, size=num_features)      
        else:
            selected_features = np.nonzero(weights)[0]

        score = np.dot(weights[selected_features], x[selected_features])
        prediction = np.sign(score)
        y_pred.append(prediction)
        if not (int(prediction) == int(y)):
            num_mistakes+=1
        if y*score <= 1:
            #compute x_hat
            x_tilde = np.zeros_like(x)
            x_tilde[selected_features] = x[selected_features]
            x_hat = []
            for i in range(len(x_tilde)):
                temp = (num_features/totol_num_features)*eps
                if not (weights[i] == 0):
                    temp = temp + (1-eps)
                x_hat.append(x_tilde[i]/temp)
            x_hat = np.array(x_hat)
            weights = weights + y * stepsize * x_hat
            weights = min(1, R/(np.linalg.norm(weights))) * weights
            weights = truncate_weights(weights, num_features)
    
    return y_pred, num_mistakes














