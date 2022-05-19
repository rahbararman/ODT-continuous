from cProfile import label
import glob, os
from pathlib import Path
import pickle
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import col
import seaborn as sns
import pickle
import numpy as np
from utils import estimate_priors_and_theta

sns.set(style="whitegrid")
def init_plotting():
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["figure.figsize"] = [20, 16]
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 22
    plt.rcParams['axes.labelsize'] = 1.8 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.8 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 1.1 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 1.8 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 1.8 * plt.rcParams['font.size']
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams.update({'figure.autolayout': True})

init_plotting()

def exp_smooth(vals, gamma=.85):
    res = [vals[0]]
    tv = res[0]
    for v in vals[1:]:
        tv = gamma * tv  + (1-gamma) * v
        res.append(tv)
    return res

methods = ['EC2', 'IG', 'US', 'efdt']
c1, c2, c3, c4 = '#d7191c', '#2b83ba', '#4dac26', '#ed9722'
cs = [c1, c2, c3, c4]
colors = {
    methods[0]:c1,
    methods[1]:c2,
    methods[2]:c3,
    'random': '#2F172E',
    'VFDT': c4,
    methods[3]: '#3f1f51'
}


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
# colors = {'EC2':'r', 'IG':'b', 'efdt':'g'}
labels = {
    'EC2':r"UFODT-$EC^2$",
    'IG': r"UFODT-$IG$",
    'US': r"UFODT-$US$",
    'efdt': "EFDT"
}

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
                plt.plot(exp_smooth(np.mean(to_plot_array, axis=0),0.7),linestyle='-', label=labels[alg], color=colors[alg])
                plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0),alpha=0.2)
            else:
                to_plot_array = results[dataset][alg]['test_perf']
                plt.plot(exp_smooth(np.mean(to_plot_array, axis=0), 0.7),linestyle='-', label=labels[alg], color=colors[alg])
                plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0),alpha=0.2)
        plt.legend()
        plt.savefig('Results/TestUtility_in_progress_num_hypo_'+str(num_hypo_in_plot)+'_'+dataset+'.pdf', format='pdf')

#Plot cost in progress
for dataset in results.keys():
    num_hypos = list(results[dataset]['IG']['numtest_progress'].keys())
    params, thetas, priors, test_csv, data_csv, theta_used_freq = estimate_priors_and_theta(dataset, rand_state=100)
    for num_hypo_in_plot in num_hypos:
        plt.clf()
        plt.xlabel('Time step')
        plt.ylabel('Cost')
        for alg in results[dataset].keys():
            if (not alg == 'efdt'):
                to_plot_array = np.array(results[dataset][alg]['numtest_progress'][num_hypo_in_plot][0]).reshape(num_rand,-1)
                plt.plot(exp_smooth(np.mean(to_plot_array, axis=0),0.7),linestyle='-', label=labels[alg], color=colors[alg])
                plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0),alpha=0.2)
            else:
                to_plot_array = [test_csv.shape[1]-1]*len(results[dataset][alg]['test_perf'][0])
                plt.plot(to_plot_array,linestyle='-', label=labels[alg], color=colors[alg])
                # plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0),alpha=0.2)
        plt.legend()
        plt.savefig('Results/Cost_in_progress_num_hypo_'+str(num_hypo_in_plot)+'_'+dataset+'.pdf', format='pdf')


#plot utility vs cost
for dataset in results.keys():
    num_hypos = list(results[dataset]['IG']['numtest_progress'].keys())
    params, thetas, priors, test_csv, data_csv, theta_used_freq = estimate_priors_and_theta(dataset, rand_state=100)
    for num_hypo_in_plot in num_hypos:
        plt.clf()
        plt.xlabel('Cost')
        plt.ylabel('Test utility')
        for alg in results[dataset].keys():
            if (not alg == 'efdt'):
                to_plot_test_utility_array = np.array(results[dataset][alg]['test_perf'][num_hypo_in_plot][0])
                to_plot_cost_array = np.array(results[dataset][alg]['numtest_progress'][num_hypo_in_plot][0])
                cost_util_dic = {}
                for (co, util) in zip(to_plot_cost_array, to_plot_test_utility_array):
                    if co in cost_util_dic:
                        cost_util_dic[co].append(util)
                    else:
                        cost_util_dic[co]=[]
                        cost_util_dic[co].append(util)
                    
                for co in cost_util_dic.keys():
                    temp = np.array(cost_util_dic[co])
                    mean = np.mean(temp)
                    std = np.std(temp)
                    cost_util_dic[co] = (mean,std)
                x = cost_util_dic.keys()
                y = [x[0] for x in cost_util_dic.values()]
                err = [x[1] for x in cost_util_dic.values()]
                plt.scatter(x, y, color=colors[alg], label=labels[alg])
                plt.errorbar(x, y, yerr=err, fmt="o", color=colors[alg])
                # plt.scatter(to_plot_test_utility_array, to_plot_cost_array, color=colors[alg], label=labels[alg])
            else:
                to_plot_array = results[dataset][alg]['test_perf'].reshape(1, -1)[0]
                
                plt.scatter(test_csv.shape[1]-1, np.mean(to_plot_array), color=colors[alg], label=labels[alg])
                plt.errorbar(test_csv.shape[1]-1, np.mean(to_plot_array), yerr=np.std(to_plot_array), fmt="o", color=colors[alg])
                
        plt.legend()
        plt.savefig('Results/TestUtility_vs_cost_num_hypo_'+str(num_hypo_in_plot)+'_'+dataset+'.pdf', format='pdf')


#plot train utility in prgoress
for dataset in results.keys():
    num_hypos = list(results[dataset]['IG']['utility_progress'].keys())
    for num_hypo_in_plot in num_hypos:
        plt.clf()
        plt.xlabel('Time step')
        plt.ylabel('Train utility')
        for alg in results[dataset].keys():
            if (not alg == 'efdt'):
                to_plot_array = np.array(results[dataset][alg]['utility_progress'][num_hypo_in_plot][0]).reshape(num_rand,-1)
                plt.plot(exp_smooth(np.mean(to_plot_array, axis=0), 0.7),linestyle='-', label=labels[alg])
                plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0),alpha=0.2)

        plt.legend()
        plt.savefig('Results/TrainUtility_in_progress_num_hypo_'+str(num_hypo_in_plot)+'_'+dataset+'.pdf', format='pdf')


#Plot cost

for dataset in results.keys():
    params, thetas, priors, test_csv, data_csv, theta_used_freq = estimate_priors_and_theta(dataset, rand_state=100)
    plt.clf()
    plt.xlabel('number of sampled hypotheses')
    plt.ylabel('cost')
    for alg in results[dataset].keys():
        if not alg == "efdt":
            numtests_mean = [np.mean(np.array(x)/len(test_csv)) for x in results[dataset][alg]['sums_all'].values()]
            numtests_std = [np.std(np.array(x)/len(test_csv)) for x in results[dataset][alg]['sums_all'].values()]
            num_samples = list(results[dataset][alg]['sums_all'].keys())
            numtests_mean.reverse()
            numtests_std.reverse()
            num_samples.reverse()
            plt.errorbar(num_samples, numtests_mean, yerr=numtests_std,linestyle='-', label=labels[alg])

        

    plt.legend()
    plt.savefig('Results/label_complexity_online_'+dataset+'.pdf', format='pdf')




#Plot train utility
for dataset in results.keys():
    params, thetas, priors, test_csv, data_csv, theta_used_freq = estimate_priors_and_theta(dataset, rand_state=100)
    plt.clf()
    plt.xlabel('number of sampled hypotheses')
    plt.ylabel('utility')
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
            plt.plot(num_samples, utility_mean, linestyle='-', label=labels[alg])
            plt.fill_between(num_samples, np.array(utility_mean)-np.array(utility_std), np.array(utility_mean)+np.array(utility_std),alpha=0.2)
    plt.ylim(0.2,0.7)
    plt.legend()
    plt.savefig('Results/Utility_online_'+dataset+'.pdf', format='pdf')



    

