from cProfile import label
import glob, os
from pathlib import Path
import pickle
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import col

from utils import estimate_priors_and_theta
pklfiles = []
for file in glob.glob("*.pkl"):
    pklfiles.append(file)

results = {}
for file in pklfiles:
    filewithoutext = Path(file).stem
    if 'efdt' in filewithoutext:
        _, _, _, dataset = filewithoutext.split('_')
        total_acc_all = pickle.load(open(file, "rb" ))
        if dataset in results.keys():
            results[dataset]['efdt'] = {}
            results[dataset]['efdt']['test_perf'] = total_acc_all
        else:
            results[dataset] = {}
            results[dataset]['efdt'] = {}
            results[dataset]['efdt']['test_perf'] = total_acc_all
    
    else:
        _, _, criterion, dataset = filewithoutext.split('_')
        [total_accuracy_progress, norm_progress, utility_progress, numtest_progress, sums_all, utility_all] = pickle.load(open(file, "rb" ))
        if dataset in results.keys():
            results[dataset][criterion] = {}
            results[dataset][criterion]['test_perf'] = total_accuracy_progress
            results[dataset][criterion]['utility_progress'] = utility_progress
            results[dataset][criterion]['numtest_progress'] = numtest_progress
            results[dataset][criterion]['sums_all'] = sums_all
            results[dataset][criterion]['utility_all'] = utility_all

        else:
            results[dataset] = {}
            results[dataset][criterion] = {}
            results[dataset][criterion]['test_perf'] = total_accuracy_progress
            results[dataset][criterion]['utility_progress'] = utility_progress
            results[dataset][criterion]['numtest_progress'] = numtest_progress
            results[dataset][criterion]['sums_all'] = sums_all
            results[dataset][criterion]['utility_all'] = utility_all


num_rand = 3
colors = {'EC2':'r', 'IG':'b', 'efdt':'g'}

#Plot test utility
for dataset in results.keys():
    num_hypos = list(results[dataset]['IG']['test_perf'].keys())
    for num_hypo_in_plot in num_hypos:
        plt.clf()
        plt.xlabel('Time step')
        plt.ylabel('Test utility')
        for alg in results[dataset].keys():
            if (not alg == 'efdt'):
                to_plot_array = np.array(results[dataset][alg]['test_perf'][num_hypo_in_plot][0]).reshape(num_rand,-1)
                plt.plot(np.mean(to_plot_array, axis=0),linestyle='-', label=alg, color=colors[alg])
                plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0),alpha=0.2)
            else:
                to_plot_array = results[dataset][alg]['test_perf']
                plt.plot(to_plot_array[1,:], label=alg, color=colors[alg])
                print(to_plot_array.shape)
        plt.legend()
        plt.savefig('Results/TestUtility_in_progress_num_hypo_'+str(num_hypo_in_plot)+'_'+dataset+'.pdf', format='pdf')
#plot train utility in prgoress
for dataset in results.keys():
    num_hypos = list(results[dataset]['IG']['utility_progress'].keys())
    for num_hypo_in_plot in num_hypos:
        plt.clf()
        plt.xlabel('Time step')
        plt.ylabel('Train Accuracy')
        for alg in results[dataset].keys():
            if (not alg == 'efdt'):
                to_plot_array = np.array(results[dataset][alg]['utility_progress'][num_hypo_in_plot][0]).reshape(num_rand,-1)
                plt.plot(np.mean(to_plot_array, axis=0),linestyle='-', label=alg)
                plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0),alpha=0.2)

        plt.legend()
        plt.savefig('Results/TrainUtility_in_progress_num_hypo_'+str(num_hypo_in_plot)+'_'+dataset+'.pdf', format='pdf')


#Plot cost

for dataset in results.keys():
    params, thetas, priors, test_csv, data_csv, theta_used_freq = estimate_priors_and_theta(dataset, rand_state=100)
    plt.clf()
    plt.xlabel('number of sampled hypotheses')
    plt.ylabel('label complexity')
    for alg in results[dataset].keys():
        if not alg == "efdt":
            numtests_mean = [np.mean(np.array(x)/len(test_csv)) for x in results[dataset][alg]['sums_all'].values()]
            numtests_std = [np.std(np.array(x)/len(test_csv)) for x in results[dataset][alg]['sums_all'].values()]
            num_samples = list(results[dataset][alg]['sums_all'].keys())
            numtests_mean.reverse()
            numtests_std.reverse()
            num_samples.reverse()
            plt.errorbar(num_samples, numtests_mean, yerr=numtests_std,linestyle='-', label=alg)

        

    plt.legend()
    plt.savefig('Results/label_complexity_online_'+dataset+'.pdf', format='pdf')




#Plot train utility
for dataset in results.keys():
    params, thetas, priors, test_csv, data_csv, theta_used_freq = estimate_priors_and_theta(dataset, rand_state=100)
    plt.clf()
    plt.xlabel('number of sampled hypotheses')
    plt.ylabel('Utility')
    for alg in results[dataset].keys():
        if not alg == "efdt":
            for k in results[dataset][alg]['utility_all'].keys():
                results[dataset][alg]['utility_all'][k] = [x[0] for x in results[dataset][alg]['utility_all'][k]]
            utility_mean = [np.mean(np.array(x)) for x in results[dataset][alg]['utility_all'].values()]
            utility_std = [np.std(np.array(x)) for x in results[dataset][alg]['utility_all'].values()]
            num_samples = list(results[dataset][alg]['utility_all'].keys())
            utility_mean.reverse()
            utility_std.reverse()
            num_samples.reverse()
            plt.plot(num_samples, utility_mean, linestyle='-', label=alg)
            plt.fill_between(num_samples, np.array(utility_mean)-np.array(utility_std), np.array(utility_mean)+np.array(utility_std),alpha=0.2)
    plt.ylim(0.2,0.7)
    plt.legend()
    plt.savefig('Results/Utility_online_'+dataset+'.pdf', format='pdf')
    

