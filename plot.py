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
    plt.rcParams['font.size'] = 37
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

methods = ['EC2', 'IG', 'US', 'efdt', 'vfdt', 'EC2-OFS', 'IG-OFS']
c1, c2, c3, c4, c5 = '#d7191c', '#2b83ba', '#4dac26', '#ed9722', '#edd222', 
cs = [c1, c2, c3, c4, c5]
colors = {
    methods[0]:c1,
    methods[1]:c2,
    methods[2]:c3,
    'random': '#2F172E',
    'VFDT': c4,
    methods[3]: '#3f1f51',
    methods[4]: c4,
    methods[5]: '#F1886D',
    methods[6]: '#63a6bd' 
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
    elif 'vfdt' in filewithoutext:
        _, _, _, dataset = filewithoutext.split('_')
        total_acc_all = pickle.load(open(file, "rb" ))
        if dataset in results.keys():
            results[dataset]['vfdt'] = {}
            results[dataset]['vfdt']['test_perf'] = total_acc_all
        else:
            results[dataset] = {}
            results[dataset]['vfdt'] = {}
            results[dataset]['vfdt']['test_perf'] = total_acc_all    
    
    elif 'OFS' in filewithoutext:
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


# colors = {'EC2':'r', 'IG':'b', 'efdt':'g'}
labels = {
    'EC2':r"UFODT-$EC^2$",
    'IG': r"UFODT-$IG$",
    'EC2-OFS':r"UFODT-$EC^2$-OFS",
    'IG-OFS': r"UFODT-$IG$-OFS",
    'US': r"UFODT-$US$",
    'efdt': "EFDT",
    'vfdt': "VFDT"
}

#Plot test utility

for dataset in results.keys():
    # for dataset in ['fetal']:
    num_hypos = list(results[dataset]['IG']['test_perf'].keys())
    ofs_num_hypo = list(results[dataset]['IG-OFS']['numtest_progress'].keys())[0]
    for num_hypo_in_plot in num_hypos:
        plt.clf()
        plt.xlabel('Time step')
        plt.ylabel('Test utility')
        for alg in results[dataset].keys():
            if (alg == 'EC2' or alg=='IG'):
                num_rand = len(list(results[dataset][alg]['sums_all'].values())[0])
                to_plot_array = np.array(results[dataset][alg]['test_perf'][num_hypo_in_plot][0]).reshape(num_rand,-1)
                plt.plot(exp_smooth(np.mean(to_plot_array, axis=0),0.7),linestyle='-', label=labels[alg], color=colors[alg])
                plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0)/np.sqrt(num_rand), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0)/np.sqrt(num_rand),alpha=0.2)
            if (alg == 'efdt' or alg=='vfdt'):
                to_plot_array = results[dataset][alg]['test_perf']
                num_rand = to_plot_array.shape[0]
                plt.plot(exp_smooth(np.mean(to_plot_array, axis=0), 0.7),linestyle='-', label=labels[alg], color=colors[alg])
                plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0)/np.sqrt(num_rand), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0)/np.sqrt(num_rand),alpha=0.2)
        #plot OFS result
        num_rand = len(list(results[dataset]['IG-OFS']['sums_all'].values())[0])
        to_plot_array = np.array(results[dataset]['IG-OFS']['test_perf'][ofs_num_hypo][0]).reshape(num_rand,-1)
        plt.plot(exp_smooth(np.mean(to_plot_array, axis=0),0.7),linestyle='-', label=labels['IG-OFS'], color=colors['IG-OFS'])
        plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0)/np.sqrt(num_rand), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0)/np.sqrt(num_rand),alpha=0.2)
        
        num_rand = len(list(results[dataset]['EC2-OFS']['sums_all'].values())[0])
        to_plot_array = np.array(results[dataset]['EC2-OFS']['test_perf'][ofs_num_hypo][0]).reshape(num_rand,-1)
        plt.plot(exp_smooth(np.mean(to_plot_array, axis=0),0.7),linestyle='-', label=labels['EC2-OFS'], color=colors['EC2-OFS'])
        plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0)/np.sqrt(num_rand), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0)/np.sqrt(num_rand),alpha=0.2)
        
        plt.legend()
        plt.savefig('Results/TestUtility_in_progress_num_hypo_'+str(num_hypo_in_plot)+'_'+dataset+'.pdf', format='pdf')

#Plot cost in progress
for dataset in results.keys():
    num_hypos = list(results[dataset]['IG']['numtest_progress'].keys())
    ofs_num_hypo = list(results[dataset]['IG-OFS']['numtest_progress'].keys())[0]
    params, thetas, priors, test_csv, data_csv, theta_used_freq = estimate_priors_and_theta(dataset, rand_state=100)
    for num_hypo_in_plot in num_hypos:
        plt.clf()
        plt.xlabel('Time step')
        plt.ylabel('Cost')
        efdt_vfdt_plotted = False
        for alg in results[dataset].keys():
            if (alg == 'EC2' or alg=='IG'):
                num_rand = len(list(results[dataset][alg]['sums_all'].values())[0])
                to_plot_array = np.array(results[dataset][alg]['numtest_progress'][num_hypo_in_plot][0]).reshape(num_rand,-1)
                plt.plot(exp_smooth(np.mean(to_plot_array, axis=0),0.7),linestyle='-', label=labels[alg], color=colors[alg])
                plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0)/np.sqrt(num_rand), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0)/np.sqrt(num_rand),alpha=0.2)
            if (alg == 'vfdt' or alg=='efdt'):
                if not efdt_vfdt_plotted:
                    efdt_vfdt_plotted = True
                    to_plot_array = [test_csv.shape[1]-1]*len(results[dataset][alg]['test_perf'][0])
                    plt.plot(to_plot_array,linestyle='-', label='VFDT/EFDT', color=c4)
                    # plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0),alpha=0.2)
        
        #plot OFS result
        num_rand = len(list(results[dataset]['IG-OFS']['sums_all'].values())[0])
        to_plot_array = np.array(results[dataset]['IG-OFS']['numtest_progress'][ofs_num_hypo][0]).reshape(num_rand,-1)
        plt.plot(exp_smooth(np.mean(to_plot_array, axis=0),0.7),linestyle='-', label=labels['IG-OFS'], color=colors['IG-OFS'])
        plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0)/np.sqrt(num_rand), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0)/np.sqrt(num_rand),alpha=0.2)
        
        num_rand = len(list(results[dataset]['EC2-OFS']['sums_all'].values())[0])
        to_plot_array = np.array(results[dataset]['EC2-OFS']['numtest_progress'][ofs_num_hypo][0]).reshape(num_rand,-1)
        plt.plot(exp_smooth(np.mean(to_plot_array, axis=0),0.7),linestyle='-', label=labels['EC2-OFS'], color=colors['EC2-OFS'])
        plt.fill_between(range(to_plot_array.shape[1]),np.mean(to_plot_array, axis=0)-np.std(to_plot_array, axis=0)/np.sqrt(num_rand), np.mean(to_plot_array, axis=0)+np.std(to_plot_array, axis=0)/np.sqrt(num_rand),alpha=0.2)

        plt.legend()
        plt.savefig('Results/Cost_in_progress_num_hypo_'+str(num_hypo_in_plot)+'_'+dataset+'.pdf', format='pdf')