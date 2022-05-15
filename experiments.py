import argparse, sys
import random

import numpy as np
from sklearn.metrics import accuracy_score
from algs import decision_tree_learning, sample_hypotheses
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier
import matplotlib.pyplot as plt

from utils import calculate_performance, calculate_total_accuracy, create_dataset_for_efdt, estimate_priors_and_theta
import pickle
#parse arguments
parser=argparse.ArgumentParser()

parser.add_argument('--rand', default=101)
parser.add_argument('--dataset', default='diabetes')
parser.add_argument('--minhypo', default=90)
parser.add_argument('--maxhypo', default=250)
parser.add_argument('--hypostep', default=10)
parser.add_argument('--alg', default='ufodt')
parser.add_argument('--thresholds', default=9)
parser.add_argument('--criterion', default='EC2')
parser.add_argument('--metric', default='accuracy')
parser.add_argument('--numrands', default=5)

args=parser.parse_args()



def main():
    num_rands = int(args.numrands)
    random_states = list(range(int(args.rand), int(args.rand)+num_rands))
    
    dataset = args.dataset
    alg = args.alg
    metric = args.metric

    if (alg == 'ufodt'):
        criterion = args.criterion
        thresholds = list(np.linspace(0.1,0.9,int(args.thresholds)))
        min_num_hypotheses = int(args.minhypo)
        max_num_hypotheses = int(args.maxhypo)
        hypotheses_step = int(args.hypostep)

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
                        if i%10 == 0:
                            print(i)
                        doc = test_csv.iloc[i].to_dict()
                        obs, y, y_hat = decision_tree_learning(thresholds,params,doc,thetas,max_steps, priors, hypothses, decision_regions, criterion, theta_used_freq)
                        sum_queries+=len(obs.items())
                        y_true.append(y)
                        y_pred.append(y_hat)
                        acc_in_progress[i].append(calculate_performance(y_true=y_true, y_pred=y_pred, metric=metric))
                        num_in_progress[i].append(len(obs.items()))
                        
                        thetas = []
                        for i in range(9):
                            thetas.append(np.random.beta(params[i][:,:,0], params[i][:,:,1]))
                        total_in_progress[i].append(calculate_total_accuracy(thetas=thetas, thresholds=thresholds, data=data_csv, priors=priors, theta_used_freq=theta_used_freq, metric=metric))
                        hypothses, decision_regions = sample_hypotheses(N=num_sampled_hypos, thetas=thetas, priors=priors, random_state=rand_state, total_samples=num_sampled_hypos, theta_used_freq=theta_used_freq)
                    accs.append(calculate_performance(y_true=y_true, y_pred=y_pred, metric=metric))

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
    
    if (alg == 'efdt'):
        efdt = ExtremelyFastDecisionTreeClassifier()
        total_acc_all = []
        for rand_state in random_states:
            X_train, X_test, y_train, y_test = create_dataset_for_efdt(dataset, rand_state)
            test_acc_in_progress = []
            for i in range(len(X_test)):
                X, y = X_test[i,:].reshape(1,-1), [y_test[i]]
                efdt.partial_fit(X, y)
                test_perf = calculate_performance(y_true=y_train, y_pred=efdt.predict(X_train), metric=metric)
                test_acc_in_progress.append(test_perf)
                
            total_acc_all.append(test_acc_in_progress)  
        
        to_save = np.array(total_acc_all)
        print(to_save.shape)
        
        f = open("efdt_test_utility_"+dataset+".pkl","wb")
        pickle.dump(to_save,f)
        f.close()

if __name__=="__main__":
    main()
