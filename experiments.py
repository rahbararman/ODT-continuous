import argparse, sys
import random

import numpy as np
from sklearn.metrics import accuracy_score
from algs import decision_tree_learning, sample_hypotheses

from utils import calculate_total_accuracy, estimate_priors_and_theta
import pickle
#parse arguments
parser=argparse.ArgumentParser()

parser.add_argument('--rand', default=101)
parser.add_argument('--dataset', default='diabetes')
parser.add_argument('--thresholds', default=9)
parser.add_argument('--criterion', default='EC2')

args=parser.parse_args()



def main():
    random_states = list(range(int(args.rand), int(args.rand)+5))
    
    dataset = args.dataset
    criterion = args.criterion
    thresholds = list(np.linspace(0.1,0.9,int(args.thresholds)))
    min_num_hypotheses = 170
    max_num_hypotheses = 200
    hypotheses_step = 10

    sums_all = {}
    utility_all = {}
    utility_progress = {}
    numtest_progress = {}
    norm_progress = {}
    total_accuracy_progress = {}

    for num_sampled_hypos in range(min_num_hypotheses, max_num_hypotheses, hypotheses_step):

        accs_all = []
        all_sum = []
        acc_in_progress = [[]]
        num_in_progress = [[]]
        norm_in_progress = [[]]
        total_in_progress = [[]]
        for rand_state in random_states:
            print('random state = '+ str(rand_state))
            params, thetas, priors, test_csv, data_csv, theta_used_freq = estimate_priors_and_theta(dataset, rand_state=rand_state) 
            if len(acc_in_progress)==1:
                acc_in_progress = acc_in_progress * len(test_csv)
                num_in_progress = num_in_progress * len(test_csv)
                norm_in_progress = norm_in_progress * len(test_csv)
                total_in_progress = total_in_progress * len(test_csv)
            hypothses, decision_regions = sample_hypotheses(N=num_sampled_hypos, thetas=thetas, priors=priors, random_state=rand_state, total_samples=num_sampled_hypos, theta_used_freq=theta_used_freq)
            print('sampled')
            accs = []
            print('Experimenting with ' + criterion)
            max_steps_values = [test_csv.shape[1]-1]
            print('max steps = '+ str(max_steps_values[0]))  
            for max_steps in max_steps_values:
                y_pred = []
                y_true = []
                sum_queries = 0 
                for i in range(len(test_csv)):
                    if i%100 == 0:
                        print(i)
                    doc = test_csv.iloc[i].to_dict()
                    obs, y, y_hat = decision_tree_learning(thresholds,params,doc,thetas,max_steps, priors, hypothses, decision_regions, criterion, theta_used_freq)
                    sum_queries+=len(obs.items())
                    y_true.append(y)
                    y_pred.append(y_hat)
                    acc_in_progress[i].append(accuracy_score(y_true, y_pred))
                    num_in_progress[i].append(len(obs.items()))
                    
                    thetas = []
                    for i in range(9):
                        thetas.append(np.random.beta(params[i][:,:,0], params[i][:,:,1]))
                    total_in_progress[i].append(calculate_total_accuracy(thetas=thetas, thresholds=thresholds, data=data_csv, priors=priors, theta_used_freq=theta_used_freq, metric='accuracy'))
                    hypothses, decision_regions = sample_hypotheses(N=num_sampled_hypos, thetas=thetas, priors=priors, random_state=rand_state, total_samples=num_sampled_hypos, theta_used_freq=theta_used_freq)
                accs.append(accuracy_score(y_true, y_pred))
            all_sum.append(sum_queries)
            accs_all.append(accs)
            print('all accuracies so far:')
            print(accs_all)
            print(all_sum)

        sums_all[num_sampled_hypos] = all_sum

        utility_all[num_sampled_hypos] = accs_all
        
        utility_progress[num_sampled_hypos] = acc_in_progress
       
        numtest_progress[num_sampled_hypos] = num_in_progress
        
        norm_progress[num_sampled_hypos] = norm_in_progress
    
        total_accuracy_progress[num_sampled_hypos] = total_in_progress

    to_save = [total_accuracy_progress, 
               norm_progress,
               utility_progress, 
               numtest_progress, 
               sums_all,  
               utility_all]
    f = open("total_dics_"+criterion+"_"+dataset+".pkl", "wb")
    pickle.dump(to_save,f)
    f.close()

if __name__=="__main__":
    main()
