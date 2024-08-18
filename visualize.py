from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import json
import os
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest


MODELS = {
    'pythia1-4b': {
        'label' : 'Pythia 1.4B',
        'size' : 1.4,
    },
    'pythia2-8b': {
        'label' : 'Pythia 2.8B',
        'size' : 2.8,
    },
    'pythia6-9b': {
        'label' : 'Pythia 6.9B',
        'size' : 6.9,
    },
    'pythia12-0b': {
        'label' : 'Pythia 12.0B',
        'size' : 12.0,
    },
    'llama7b': {
        'label' : 'Llama 7B',
        'size' : 7.0,
    }, 
    'llama13b': {
        'label' : 'Llama 13B',
        'size' : 13.0,
    }, 
    'llama30b': {
        'label' : 'Llama 30B',
        'size' : 30.0,
    }
}

LOSSES = {
    'unaligned': {
        'label' : 'unaligned',
    },
    'sft': {
        'label' : 'SFT',
    }, 
    'csft': {
        'label' : 'CSFT'
    },
    'ppo': {
        'label' : 'offline PPO'
    }, 
    'dpo' : {
        'label' : 'DPO',
    },
    'slic' : {
        'label' : 'SLiC',
    },
    'kto' : {
        'label' : 'KTO',
    },
    'sft+csft': {
        'label' : 'SFT+CSFT'
    },
    'sft+ppo': {
        'label' : 'SFT+offline PPO'
    }, 
    'sft+dpo' : {
        'label' : 'SFT+DPO',
    },
    'sft+slic' : {
        'label' : 'SFT+SLiC',
    },
    'sft+kto' : {
        'label' : 'SFT+KTO',
    },
}


def process_archangel(fn='assets/results.jsonl'):
    results = defaultdict(lambda: defaultdict(dict))

    for line in open(fn):
        result = json.loads(line.strip())
        
        if result['exp_name'].startswith('archangel'):
            suite, loss, model = result['exp_name'].split('_')

            if result['baseline']['name'] == 'chosen' and result['candidate']['name'] == 'policy':
                results[loss][model] = result

    return results


def process_fracdata(fn='assets/results.jsonl'):
    """
    Returns a nested dict of losses, models, and 'F{U|D}{fraction of (un)desirable examples kept'.
    Only rows whose exp_name starts with fracdata are kept.
    """
    results = defaultdict(lambda: defaultdict(dict))

    for line in open(fn):
        result = json.loads(line.strip())

        if result['exp_name'].startswith('fracdata'):
            suite, loss, model, desirable_frac, desirable_weight, undesirable_frac, undesirable_weight = result['exp_name'].split('_')
            
            if result['baseline']['name'] == 'chosen' and result['candidate']['name'] == 'policy':
                results[loss][model][desirable_frac] = result
                results[loss][model][undesirable_frac] = result

    return results


def plot_policy_lengths(model, losses, fn):
    results = process_archangel()
    # Plotting the histogram
    plt.figure(figsize=(12, 6))

    data = [ results[l][model]['candidate']['lengths'] for l in losses ]
    plt.boxplot(data, vert=True, patch_artist=True)

    # Add labels and title
    plt.xticks([i+1 for i in range(len(losses))], [ LOSSES[l]['label'] for l in losses], fontsize=18)
    plt.ylabel(f"Output Lengths ({MODELS[model]['label']})", fontsize=20)

    plt.tight_layout()
    # Adding legend
    plt.savefig(f'assets/figures/{fn}.png')


def plot_model_winrates(losses, fn, confidence=0.90):
    results = process_archangel()
    # Plotting the histogram
    fig, ax = plt.subplots(figsize=(18, 6))
    # Set the bar width
    bar_width = 0.1
    
    red_palette = plt.cm.Reds(np.linspace(0.1, 0.9, 7))
    blue_palette = plt.cm.Blues(np.linspace(0.1, 0.9, 7))

    # Set the positions for the bars
    positions = np.arange(len(losses))

    for i in range(len(MODELS)):
        m = list(MODELS.keys())[i]
        ci = [ binomtest(k=results[l][m]['candidate']['wins'], n=results[l][m]['total'], p=0.5).proportion_ci(confidence) for l in losses ]
        yerr = [ (c[1] - c[0]) / 2 for c in ci ]
        winrates = [( results[l][m]['candidate']['wins'] / results[l][m]['total']) - 0.5 for l in losses ]

        for j in range(len(winrates)):
            color_palette = red_palette if m.startswith('pythia') else blue_palette
            bar = plt.bar(
                (positions[j] - bar_width * len(losses) / 2 + bar_width * i), 
                winrates[j],
                color=color_palette[3], 
                yerr=yerr[j],
                width=bar_width,
                error_kw=dict(lw=1, capsize=1, capthick=0.5, ecolor=color_palette[5]),
            )

    # Add labels and title
    plt.ylabel('Win Rate - 0.5', fontsize=18)
    plt.title('Does the aligned model beat the SFT target?', fontsize=20)
    plt.xticks([pos + bar_width / (len(MODELS) - 1) for pos in positions], [ LOSSES[l]['label'] for l in losses ], fontsize=18)

    ax.legend(handles=[Patch(facecolor=red_palette[3], label='pythia-{1.4B, 2.8B, 6.9B, 12.0B}'),
                       Patch(facecolor=blue_palette[3], label='llama-{7B, 13B, 30B}')], loc='best', fontsize=14)
    
    plt.tight_layout()
    # Adding legend
    plt.savefig(f'assets/figures/{fn}.png')


def plot_data_win_curves(loss='kto', model='llama7b', confidence=0.90):
    results_fracdata = process_fracdata()
    results_archangel = process_archangel()

    frac = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    winrates_des, ci_des = [], []
    winrates_und, ci_und = [], []

    for x in frac[:-1]:
        wins_des = results_fracdata[loss][model][f'FD{x}']['candidate']['wins']
        total_des = results_fracdata[loss][model][f'FD{x}']['total']
        ci = binomtest(k=wins_des, n=total_des, p=0.5).proportion_ci(confidence)
        ci_des.append((ci[1] - ci[0]) / 2)
        winrates_des.append(wins_des / total_des)

        wins_und = results_fracdata[loss][model][f'FU{x}']['candidate']['wins']
        total_und = results_fracdata[loss][model][f'FU{x}']['total']
        ci = binomtest(k=wins_und, n=total_und, p=0.5).proportion_ci(confidence)
        ci_und.append((ci[1] - ci[0]) / 2)
        winrates_und.append(wins_und / total_und)

    wins_all_data = results_fracdata[loss][model][f'FU1.0']['candidate']['wins']
    total_all_data = results_fracdata[loss][model][f'FU1.0']['total']
    ci = binomtest(k=wins_all_data, n=total_all_data, p=0.5).proportion_ci(confidence)
    ci_des.append((ci[1] - ci[0]) / 2)
    ci_und.append((ci[1] - ci[0]) / 2)
    winrates_des.append(wins_all_data / total_all_data)
    winrates_und.append(wins_all_data / total_all_data)

    dpo_winrate = (results_archangel['dpo'][model]['candidate']['wins'] / results_archangel['dpo'][model]['total'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    ax1.plot(frac, winrates_des, linestyle='dotted', marker='o', color='r')
    ax1.axhline(y=dpo_winrate, color='k', linestyle='--')
    ax1.set_xlabel('Fraction of Desirable Data Kept')
    ax2.plot(frac, winrates_und, linestyle='dotted', marker='o', color='b', label='Desirable Fraction')
    ax2.axhline(y=dpo_winrate, color='k', linestyle='--')
    ax2.text(0.6, dpo_winrate - 0.02, 'DPO-Aligned Llama7B', color='k', ha='center')
    ax2.set_xlabel('Fraction of Undesirable Data Kept')
   
    ax1.set_ylim([0.2, 0.5])
    fig.suptitle('Winrate (KTO-Aligned Llama7B vs. SFT Target)') 

    plt.tight_layout()
    # Adding legend
    plt.savefig(f'assets/figures/fracdata.png')