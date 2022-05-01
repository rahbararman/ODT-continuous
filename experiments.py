import argparse, sys
import random

from defusedxml import DTDForbidden
import numpy as np
from sklearn.metrics import accuracy_score
from algs import decision_tree_learning, sample_hypotheses

from utils import calculate_total_accuracy, estimate_priors_and_theta

parser=argparse.ArgumentParser()

parser.add_argument('--rand', default=101)
parser.add_argument('--dataset', default='20newsgroup')
parser.add_argument('--thresholds', default=9)

args=parser.parse_args()



def main():
    random_states = list(range(int(args.rand), int(args.rand)+5))
    
    #EC2 continuous
    dataset = args.dataset
    thresholds = list(np.linspace(0.1,0.9,int(args.thresholds)))
    min_num_hypotheses = 100
    max_num_hypotheses = 1000
    hypotheses_step = 100

    sums_all_EC2 = {}
    utility_all_EC2 = {}
    utility_progress_EC2 = {}
    numtest_progress_EC2 = {}
    norm_progress_EC2 = {}
    total_accuracy_progress_EC2 = {}

    for num_sampled_hypos in range(min_num_hypotheses, max_num_hypotheses, hypotheses_step):

        accs_all = []
        all_sum = []
        acc_in_progress = [[]]
        num_in_progress = [[]]
        norm_in_progress = [[]]
        total_in_progress = [[]]
        for rand_state in random_states:
            print('random state = '+ str(rand_state))
            params, thetas, priors, test_csv, data_csv = estimate_priors_and_theta(dataset, rand_state=rand_state) 
            if len(acc_in_progress)==1:
                acc_in_progress = acc_in_progress * len(test_csv)
                num_in_progress = num_in_progress * len(test_csv)
                norm_in_progress = norm_in_progress * len(test_csv)
                total_in_progress = total_in_progress * len(test_csv)
            hypothses, decision_regions = sample_hypotheses(N=num_sampled_hypos, thetas=thetas, priors=priors, random_state=rand_state, total_samples=num_sampled_hypos)
            print('sampled')
            accs = []
            print('Experimenting with EC2')
            max_steps_values = [test_csv.shape[1]-1]
            print('max steps = '+ str(max_steps_values[0]))  
            for max_steps in max_steps_values:
                y_pred = []
                y_true = []
                sum_queries = 0 
                for i in range(len(test_csv)):
                    if i%1 == 0:
                        print(i)
                    doc = test_csv.iloc[i].to_dict()
                    obs, y, y_hat = decision_tree_learning(thresholds,params,doc,thetas,max_steps, priors, hypothses, decision_regions, 'EC2')
                    sum_queries+=len(obs.items())
                    y_true.append(y)
                    y_pred.append(y_hat)
                    acc_in_progress[i].append(accuracy_score(y_true, y_pred))
                    num_in_progress[i].append(len(obs.items()))
                    
                    thetas = []
                    for i in range(9):
                        thetas.append(np.random.beta(params[i][:,:,0], params[i][:,:,1]))
                    total_in_progress[i].append(calculate_total_accuracy(thetas=thetas, thresholds=thresholds, data=data_csv, priors=priors, metric='accuracy'))
                    hypothses, decision_regions = sample_hypotheses(N=num_sampled_hypos, thetas=thetas, priors=priors, random_state=rand_state, total_samples=num_sampled_hypos)
                accs.append(accuracy_score(y_true, y_pred))
            all_sum.append(sum_queries)
            accs_all.append(accs)
            print('all accuracies so far:')
            print(accs_all)
            print(all_sum)

        sums_all_EC2[num_sampled_hypos] = all_sum

        utility_all_EC2[num_sampled_hypos] = accs_all
        
        utility_progress_EC2[num_sampled_hypos] = acc_in_progress
       
        numtest_progress_EC2[num_sampled_hypos] = num_in_progress
        
        norm_progress_EC2[num_sampled_hypos] = norm_in_progress
    
        total_accuracy_progress_EC2[num_sampled_hypos] = total_in_progress

    import pickle
    to_save = [total_accuracy_progress_EC2, 
               norm_progress_EC2,
               utility_progress_EC2, 
               numtest_progress_EC2, 
               sums_all_EC2,  
               utility_all_EC2]
    f = open("total_dics_EC2_"+dataset+".pkl", "wb")
    pickle.dump(to_save,f)
    f.close()
    




if __name__=="__main__":
    main()
