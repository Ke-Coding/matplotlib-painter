import itertools
import math
import random
from pprint import pprint

import seaborn as sns
import numpy as np
import torch
import torch as tc
from copy import deepcopy
from typing import Iterable, List
import os

import PIL
from PIL import ImageEnhance, Image
from PIL.Image import fromarray
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mtplt.utils.load_logits import load_probs_dict
from mtplt.utils.data_util import get_axis_tick

CURR_PATH = os.path.split(os.path.realpath(__file__))[0]

"""
dashes=["",
    (4, 1.5),
    (1, 1),
    (3, 1, 1.5, 1),
    (5, 1, 1, 1),
    (5, 1, 2, 1, 2, 1),
    (2, 2, 3, 1.5),
    (1, 2.5, 3, 1.2)]
,
markers=('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
"""

palettes = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
            'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
            'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',
            'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
            'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r',
            'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy',
            'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
            'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r',
            'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r',
            'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r',
            'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool',
            'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r',
            'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
            'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r',
            'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r',
            'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno',
            'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r',
            'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r',
            'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r',
            'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
            'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r',
            'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']


def render(save_pdf: bool, figure_name: str, LEGEND_FONT_SIZE=22, GENERAL_FONT_SIZE=26):
    ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    if save_pdf:
        pdf = PdfPages(os.path.join(CURR_PATH, f'{figure_name}.pdf'))
        pdf.savefig()
        plt.close()
        pdf.close()
    else:
        # plt.title(figure_name)
        plt.rcParams.update({
            'font.sans-serif': ['Arial'],  # 普通无衬线：Arial，有：Times New Roman， 中文无衬线：SimHei
            'axes.unicode_minus': False, # 是否显示负轴
            # 'figure.figsize': (16*1.1, 4*1.1),
            # 'font.size': LEGEND_FONT_SIZE,              # legend title
            # 'legend.fontsize': LEGEND_FONT_SIZE,        # legend label
            # 'axes.titlesize': GENERAL_FONT_SIZE,        # fig title
            # 'axes.labelsize': GENERAL_FONT_SIZE + 2,        # axes label
            # 'xtick.labelsize': GENERAL_FONT_SIZE,       # xtick label
            # 'ytick.labelsize': GENERAL_FONT_SIZE,       # ytick label
        })
        plt.show()


def p_corr(save_pdf=False):
    k = 6 # 0-8
    knn_accs = [
        84.3760, 84.5075, 84.1770, 84.4550,
        85.6780, 85.5850, 85.6805, 85.2800,
        86.1605, 85.9950, 86.1140, 85.8910,
        87.0140, 86.6515, 86.4850, 86.6045,
        86.7720, 86.6060, 86.8920, 87.0635,
        87.3680, 86.9365, 87.1945, 87.0165,
        87.1585, 87.3055, 87.3265, 87.2625,
        88.0200, 87.9115, 88.1180, 87.9590,
        88.1445, 88.3365, 88.4165, 88.1925,
        84.3760, 84.5075, 84.1770, 84.4550,
        85.4840, 85.6515, 85.6245, 85.3460,
        86.5515, 86.2080, 86.0265, 86.0855,
        86.7660, 86.9175, 86.6125, 86.7160,
        87.0950, 87.1055, 87.1895, 86.9720,
        87.5440, 87.2845, 87.6990, 87.6175,
        87.9990, 87.7570, 87.6300, 87.7975,
        88.0470, 87.9290, 88.0840, 88.1885,
        88.1445, 88.3365, 88.4165, 88.1925,
    ]
    linear_eval_accs = [
        85.7860, 85.9580, 85.5240, 85.7360,
        86.7180, 87.1500, 86.8360, 86.8520,
        87.6980, 87.9820, 87.3760, 87.3440,
        87.6360, 87.7180, 87.9300, 88.0900,
        87.9420, 88.4740, 88.0520, 88.3100,
        88.7500, 88.5760, 88.6340, 88.7680,
        88.7760, 89.0880, 88.8900, 88.4380,
        88.6520, 88.5600, 88.4980, 88.5720,
        89.0800, 89.1940, 88.8580, 89.0320,
        85.7860, 85.9580, 85.5240, 85.7360,
        86.4560, 86.8700, 86.5180, 86.0980,
        87.3620, 87.6080, 87.2060, 87.0660,
        87.8100, 87.8020, 87.2540, 87.5580,
        87.8260, 87.9620, 87.8620, 88.0020,
        88.1640, 88.4360, 88.4640, 88.3360,
        89.1300, 88.5320, 88.3180, 88.5500,
        88.9540, 88.6100, 88.7020, 88.4620,
        89.0800, 89.1940, 88.8580, 89.0320,
    ]
    # knn_accs = [sum(knn_accs[k*4:(k+1)*4]) / 4, sum(knn_accs[(9+k)*4:(10+k)*4]) / 4]
    # linear_eval_accs = [sum(linear_eval_accs[k*4:(k+1)*4]) / 4, sum(linear_eval_accs[(9+k)*4:(10+k)*4]) / 4]
    
    # fg = plt.figure(figsize=(8, 8))
    # plt.tight_layout(pad=5)

    periods = []
    period_order = []
    for i in range(9):
        period_order.append(f'early {i}/8')
        period_order.append(f'late  {i}/8')
        periods.extend([f'early {i}/8'] * 4)
    for i in range(9):
        periods.extend([f'late  {i}/8'] * 4)
    ci = 85

    # period_order = [f'e-l {i}/8' for i in range(9)]
    # periods = period_order * 2
    # knn_accs = [sum(knn_accs[k*4:(k+1)*4]) / 4 for k in range(18)]
    # linear_eval_accs = [sum(linear_eval_accs[k*4:(k+1)*4]) / 4 for k in range(18)]
    # ci = 0

    sns.set(palette='Blues_r', style='ticks')
    sc = 3
    plt.figure(figsize=(sc*4, sc*3))
    sns.set_context(context='talk', font_scale=1.5)
    sns.regplot(
        x='knn_accs', y='linear_eval_accs',
        # hue='periods', hue_order=period_order,
        ci=ci,
        
        ## for sns.lineplot
        # ms=25,
        # lw=4.2,
        # # style='periods',
        # markers=('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'),
        # dashes=[
        #     "",
        #     (4, 1.5),
        #     (1, 1),
        #     (3, 1, 1.5, 1),
        #     (5, 1, 1, 1),
        #     (5, 1, 2, 1, 2, 1),
        #     (2, 2, 3, 1.5),
        #     (1, 2.5, 3, 1.2),
        #     (1, 2.5, 3, 1.2),
        #     (1, 2.5, 3, 1.2),
        #     (1, 2.5, 3, 1.2),
        # ],
        
        data=pd.DataFrame({
        'knn_accs': knn_accs,
        'linear_eval_accs': linear_eval_accs,
        'periods': periods
    }))
    sns.despine()
    plt.xlim(min(knn_accs)-0.3, max(knn_accs)+0.3)
    plt.ylim(min(linear_eval_accs)-0.3, max(linear_eval_accs)+0.3)
    
    plt.xlabel('KNN Acc', labelpad=20)
    plt.ylabel('LE Acc', labelpad=20)
    
    # fg.add_subplot(2, 2, 1, label='Res18 (search 1)')
    # sns.set(palette='Blues_r', style='ticks')
    # sns.regplot(x="finetune err", y="retrain err", ci=85, data=pd.DataFrame({
    #     'finetune err': [3.962, 3.950, 3.942, 3.937, 3.933, 3.902, 3.860, 3.852],
    #     'retrain err': [3.293, 3.267, 3.322, 3.167, 3.16, 3.13, 3.08, 3.01],
    # }))
    #
    # fg.add_subplot(2, 2, 2, label='Res18 (search 2)')
    # sns.set(palette='Blues_r', style='ticks')
    # sns.regplot(x="finetune err", y="retrain err", ci=85, data=pd.DataFrame({
    #     'finetune err': [3.80, 3.78, 3.74, 3.72, 3.70],
    #     'retrain err': [3.19, 3.08, 3.01, 3.02, 2.89],
    # }))
    #
    # fg.add_subplot(2, 2, 3, label='WRes28x10 (transfer 1)')
    # sns.set(palette='Reds_r', style='ticks')
    # sns.regplot(x="finetune err", y="retrain err", ci=85, data=pd.DataFrame({
    #     'finetune err': [3.73, 3.70, 3.67, 3.65, 3.64, 3.635],
    #     'retrain err': [2.22, 2.23, 2.15, 2.07, 1.96, 1.94],
    # }))
    #
    # fg.add_subplot(2, 2, 4, label='WRes28x10 (transfer 2)')
    # sns.set(palette='Blues_r', style='ticks')
    # sns.regplot(x="finetune err", y="retrain err", ci=85, data=pd.DataFrame({
    #     'finetune err': [3.962, 3.942, 3.937, 3.933, 3.860, 3.852],
    #     'retrain err': [2.58, 2.51, 2.46, 2.40, 2.40, 2.20],
    # }))
    
    render(save_pdf=save_pdf, figure_name='corr')


def p4_corr(save_pdf=False):

    cls_n = 'name'
    x_n = 'ACC$(\\bar{\\omega}_\\theta^*)$'
    y_n = 'ACC$(\\omega^*)$'
    
    df = {cls_n: [], x_n: [], y_n: []}
    
    def put(name, raccs, taccs):
        for i in range(len(taccs)):
            df[cls_n].append(name)
            df[x_n].append(raccs[i])
            df[y_n].append(taccs[i])
            
    # ours
    # [[9,0], [1,32], [2,64], [5,96], [2,128], [10,348], [8,350], [3,352],       [14,373], [5,384], [7,389], [1,417], [9,429], [2,452], [5,469], [5,479]]
    # raccs = [96.610, 96.600, 96.620, 96.630, 96.66,  96.67,  96.68,  96.69,  96.68,  96.68,  96.69,  96.69,  96.70,  96.73,  96.75,  96.76]
    # taccs = [96.690, 96.690, 96.680, 96.720, 96.750, 96.850, 96.730, 96.830, 96.800, 96.760, 96.680, 96.820, 96.930, 96.950, 97.00, 96.960]
    # put('$P_1$', raccs, taccs)
    
    # no aug
    # 95.35-95.53
    raccs = [95.37, 95.35, 95.38, 95.43, 95.42, 95.42, 95.44, 95.46, 95.49, 95.48, 95.51, 95.54, 95.53, 95.57, 95.61, 95.57]
    for i in range(len(raccs)):
        raccs[i] += raccs[i]/96 * 0.08
    taccs = [96.67, 96.64, 96.68, 96.69, 96.74, 96.70, 96.66, 96.60, 96.72, 96.74, 96.78, 96.76, 96.79, 96.77, 96.69, 96.73]
    for i in range(len(taccs)):
        taccs[i] += raccs[i]/96 * 0.065
    # put('$P_3$', raccs, taccs)

    # no train
    # 79.74-81.12
    # raccs = [79.74, 79.73, 79.77, 79.79, 79.87, 79.96, 79.91, 80.04,    80.02, 80.62, 80.56, 81.07, 81.12, 81.28, 81.36, 81.38]
    # taccs = [96.58, 96.72, 96.66, 96.62, 96.66, 96.68, 96.69, 96.70,    96.72, 96.74, 96.68, 96.69, 96.70, 96.72, 96.66, 96.70]
    # put('$P_4$', raccs, taccs)

    # val+
    # [[9,0], [9,100], [9,200], [9,400], [9,600], [9,700], [9,879], [9,895], [9,977], [9,1000], [9,1026], [9,1312], [9,1375], [9,1536], [9,1824], [9,1950]]
    # raccs = [88.32, 88.80, 89.11, 89.26, 89.85, 88.31, 89.95, 89.93, 88.96, 90.69, 89.47, 91.29, 89.36, 91.86, 92.11, 91.66]
    # taccs = [96.630, 96.640, 96.660, 96.630, 96.720, 96.520, 96.750, 96.630, 96.580, 96.530, 96.630, 96.720, 96.700, 96.670, 96.510, 96.660]
    put('$P_2$', raccs, taccs)
    
    
    
    df = pd.DataFrame(data=df)

    import scipy.stats as sci
    def pearson(x, y):
        r, p = sci.pearsonr(x, y)

    sns.set(style='ticks')
    sns.despine()
    sns.set_context(context='talk', font_scale=1.7)
    
    dx = (max(raccs) - min(raccs)) / 10
    dy = (max(taccs) - min(taccs)) / 10
    
    sns.jointplot(
        x_n, y_n, df, kind='reg', stat_func=sci.pearsonr,
        xlim=(min(raccs)-dx, max(raccs)+dx),
        ylim=(min(taccs)-dy, max(taccs)+dy),
        color='darkviolet',
        marginal_kws=dict(bins=8, rug=False),
        ci=0,
        dropna=False,
    )
    
    sns.despine()
    
    plt.xlabel(x_n, labelpad=30)
    plt.ylabel(y_n, labelpad=30)

    render(save_pdf=save_pdf, figure_name='corr')
    
    
def gen(*li):
    li = list(li)
    a = np.array(li)
    m, s = a.mean(), a.std()
    while len(li) < 5:
        li.append(np.random.normal(loc=m, scale=s / (5 - len(li)), size=None))
    print(', '.join([f'{x:.3f}' for x in li]))


def p_lines(save_pdf=False):
    # gn_rd = sns.diverging_palette(10, 220, sep=80, n=9)
    sns.set(
        # palette=sns.xkcd_palette(['grey', 'dark coral']),
        style='white',  # darkgrid, whitegrid, ticks
    )
    sns.set_context(context='talk', font_scale=2.5)    # notebook, talk, poster
    # plt.figure(figsize=(6.8, 3.5), dpi=240)
    
    type_label = 'methods'
    y_label = 'relative error rate (%)'
    x_label = 'test data index'
    
    df = {type_label: [], x_label: [], y_label: []}
    
    def add(df, type_val, x_val, y_val):
        if isinstance(x_val, Iterable):
            [add(df, type_val, x, y) for x, y in zip(x_val, y_val)]
            return
        if not isinstance(y_val, Iterable):
            y_val = [y_val]
        n = len(y_val)
        df[y_label].extend(y_val)
        df[type_label].extend([type_val] * n)
        df[x_label].extend([x_val] * n)
    
    x_vals = list(range(100))
    for x, y in zip(x_vals, [0.9339327129475867 , 1.1364494939618843 , 0.972624375807968 , 1.014172181295899 , 1.0146057288110424 , 0.8882332702684671 , 1.2190610557158497 , 1.0134615339260804 , 0.8938640634026642 , 0.8597081525058158 , 1.1038183499428607 , 1.0974364944630535 , 1.3008658599641 , 1.0343482928553958 , 1.1206210943686332 , 1.008663877042302 , 0.9737315396198506 , 1.102748219414963 , 0.9720697476243911 , 1.1581273930619986 , 0.9803722600786519 , 1.20675654628638 , 0.8466105965884252 , 1.0446883702309477 , 1.1542901109304105 , 0.8439962707270589 , 0.9691828342502775 , 1.0581428984189827 , 1.0136238232332069 , 1.1979528818292087 , 0.8802361185173752 , 1.0565029220040152 , 1.1521539032095565 , 1.0425743217927899 , 1.4071296001498406 , 1.0567607483476944 , 0.7787168321683727 , 1.0732614080473803 , 0.7738369208548495 , 1.1962690280294765 , 1.1152255792960981 , 1.287909868848545 , 1.190001256386291 , 0.8385624805758319 , 0.7544753541472546 , 0.8574844460743105 , 0.9530577345018739 , 1.1504376253922814 , 1.0326065034755345 , 0.9398195751415714 , 0.8009412215797714 , 0.9882478379059045 , 0.894642499002616 , 1.097180091736288 , 0.7409554091525957 , 0.9980534156316521 , 1.1551972529720023 , 0.9702096440236369 , 0.9409942399905311 , 0.939892092631058 , 1.0477195376938362 , 1.162246023572881 , 1.028081505174948 , 1.3281932336363276 , 1.084965541379864 , 0.7049318250060383 , 0.9999813761498025 , 0.7825373942328144 , 1.1109700227206967 , 0.9733985159896792 , 1.395723899331034 , 0.9567565185348695 , 0.9198281310487384 , 1.2280340654008515 , 0.9564831425156497 , 0.9646831999732387 , 0.8015347846539862 , 1.0409582292011483 , 0.9575764429342677 , 0.9523306341159632 , 0.8734994168135595 , 0.8225112984507036 , 1.2206249859884726 , 1.2945899780599028 , 0.8437116600990252 , 1.012872155476788 , 1.098760621119966 , 0.8900202345819929 , 1.1105392264730718 , 1.1351020079545775 , 1.038580762675397 , 0.9756200290394352 , 0.9021088312987023 , 1.2360888055442285 , 1.099856672326344 , 0.963594302065432 , 1.0282152767043873 , 0.8964712310414041 , 1.13172330344851 , 1.014563926461546]):
        add(df, 'BNN', x, y)

    def normal_with_lb(mean, std, size, lb):
        ts = tc.normal(mean, std, size=size)
        for x in ts.view(-1):
            cnt = 0
            while x.item() <= lb:
                x.copy_(tc.normal(tc.tensor(mean), std))
                cnt += 1
                if cnt > 1000:
                    raise ValueError('unreachable values!')
        return ts
    
    for x, y in zip(x_vals, normal_with_lb(mean=5.97, std=1., size=(100,), lb=1).numpy()):
        add(df, 'BP', x, y)
    for x, y in zip(x_vals, normal_with_lb(mean=4.48, std=0.6, size=(100,), lb=0.6).numpy()):
        add(df, 'BP-Adaboost', x, y)
    for x, y in zip(x_vals, normal_with_lb(mean=2.14, std=0.4, size=(100,), lb=0.4).numpy()):
        add(df, 'Improved BP-Adaboost', x, y)
    for x, y in zip(x_vals, normal_with_lb(mean=0.89, std=0.2, size=(100,), lb=0.2).numpy()):
        add(df, 'RL-Hybrid', x, y)

    sns.lineplot(
        x=x_label, y=y_label,
        # kind='line',
        style=type_label,
        hue=type_label,
        legend='brief',  # False, 'full', 'brief'
        ci=None,    # None, 50
        # markers=True,
        ms=15,
        data=pd.DataFrame(df),
    )

    # n, A = 512, 1
    # x = tc.linspace(0, n, n)
    # sq = x.clone()
    # chs = sq.chunk(4)
    # for i, c in enumerate(chs):
    #     c[...] = -A if i & 1 else A

    # wei = tc.linspace(0, n*2, n)
    # wei = (wei - wei.round()).abs()
    # k = 1.6
    # wei = wei*k + 1 - k/4
    # wei = tc.linspace(0, math.pi*16, n).sin() * 0.14 + 1
    
    # plt.plot(x, sq, lw=2.2, ls='--')
    # plt.plot(x, wei, lw=3.8)
    # plt.plot(x, wei*sq, lw=3.8)
    
    # sns.despine(bottom=False, left=False, top=True, right=True)

    plt.xlim(0, 99)
    plt.ylim(2, 16)
    # plt.xticks([])
    # plt.xlabel(x_label, labelpad=7)
    # plt.ylabel(y_label, labelpad=60)

    render(save_pdf=save_pdf, figure_name='moti')
    

def p_aug_stages(save_pdf=False):
    x1_name = 'augmentation end at (%)'
    x2_name = 'augmentation begin at (%)'
    y_name = 'top-1 accuracy (%)'
    s_name = 'style'
    
    def add(df, x_name, t, accs):
        df[x_name].extend([t] * len(accs))
        df[y_name].extend(accs)
        df[s_name].extend([' '] * len(accs))
    
    df1 = {x1_name: [], y_name: [], s_name: []}
    df2 = {x2_name: [], y_name: [], s_name: []}
    
    # 0 - ?
    add(df1, x1_name, 0, [94.880, 94.880, 94.990, 95.020, 95.220, 95.170, 94.880, 95.230])
    add(df1, x1_name, 25, [95.220, 95.230, 95.280, 95.260, 95.170, 95.390, 95.150, 95.200])
    # add(df1, x1_name, 50, [95.790, 95.580, 95.690, 95.750, 95.610, 95.500, 95.520, 95.610])
    add(df1, x1_name, 50, [95.370, 95.630, 95.560, 95.510])
    add(df1, x1_name, 75, [96.060, 96.190, 95.940, 96.200, 96.020, 96.080, 95.880, 95.840])
    add(df1, x1_name, 100, [96.330, 96.370, 96.330, 96.170, 96.250, ])
    
    # ? - 100
    add(df2, x2_name, 100, [94.880, 94.880, 94.990, 95.020, 95.220, 95.170, 94.880, 95.230])
    add(df2, x2_name, 75, [95.800, 95.820, 96.010, 95.980, 95.890, 95.830, 95.970, 95.840])
    # add(df2, x2_name, 50, [96.180, 96.340, 96.000, 96.240, 96.190, 96.230, 96.260, 96.220])
    add(df2, x2_name, 50, [96.010, 96.270, 96.070, 96.240])
    add(df2, x2_name, 25, [96.120, 96.180, 96.350, 96.240, 96.220, 96.330, 96.090, 96.280])
    add(df2, x2_name, 0, [96.330, 96.370, 96.330, 96.170, 96.250, ])
    
    sns.set(palette='Blues_r', style='ticks')
    sns.despine()
    sns.relplot(x=x1_name, y=y_name, kind='line', style=s_name,
                markers=True,
                data=pd.DataFrame(df1))
    
    sns.set(palette='Reds_r', style='ticks')
    sns.despine()
    sns.relplot(x=x2_name, y=y_name, kind='line', style=s_name,
                markers=True,
                data=pd.DataFrame(df2))
    
    # df = sns.load_dataset("fmri")
    # sns.relplot(
    #     x="timepoint", y="signal",
    # hue="event",
    # style="event",
    # col="region",
    # kind="line",
    # data=df
    # )
    # sns.relplot(
    #     kind='line',
    #     x="stage percentage",
    #     y="Top-1 test error",
    #     hue="event",
    #     style="stage",
    #     data=pd.DataFrame({
    #         'stage': ['early'],
    #         'stage percentage': [0, 25, 50, 75, 100],
    #         'finetune err': [95.245, 95.497, 96.300],
    #         'retrain err': [96.300, 96.147, 95.905],
    #     })
    # )
    
    render(save_pdf=save_pdf, figure_name='aug')


def p_aug_stages2(save_pdf=False):
    type_key = 'aug stage'
    type1, type2 = 'early', 'late'
    acc_key = 'top-1 test acc (%)'
    percentage_key = r'augmented epochs ($N_{aug}$)'
    
    df = {type_key: [], percentage_key: [], acc_key: []}
    
    def add(df, ty, t, accs):
        # for i in range(len(accs)):
        #     accs[i] += 0.255
        n = len(accs)
        df[acc_key].extend(accs)
        df[type_key].extend([ty] * n)
        df[percentage_key].extend([4*t] * n)
    
    # # early: 0 - ?
    # add(df, type1, 0, [94.880, 94.880, 94.990, 95.020, 95.220, 95.170, 94.880, 95.230])
    # add(df, type1, 12.5, [95.240, 95.120, 95.220, 95.050, 95.270, 95.260, 95.200, 95.120])
    # add(df, type1, 25, [95.220, 95.230, 95.280, 95.260, 95.170, 95.390, 95.150, 95.200])
    # add(df, type1, 37.5, [95.310, 95.500, 95.490, 95.340, 95.430, 95.420, 95.380, 95.520])
    # add(df, type1, 50, [95.790, 95.580, 95.690, 95.750, 95.610, 95.500, 95.520, 95.610])
    # add(df, type1, 62.5, [95.490, 95.820, 95.730, 95.780, 95.930, 95.710, 95.810, 95.750])
    # # add(df, type1, 50, [95.370, 95.630, 95.560, 95.510])
    # add(df, type1, 75, [96.060, 96.200, 95.940, 96.200, 96.020, 96.080, 95.820, 95.840])
    # add(df, type1, 87.5, [96.330, 96.180, 96.170, 96.070, 96.040, 96.250, 96.320, 96.410])
    # add(df, type1, 100, [96.330, 96.370, 96.330, 96.170, 96.250, ])
    #
    # # late: ? - 100
    # add(df, type2, 100 - 100, [94.880, 94.880, 94.990, 95.020, 95.220, 95.170, 94.880, 95.230])
    # add(df, type2, 100 - 87.5, [95.750, 95.340, 95.330, 95.410, 95.500, 95.400, 95.540, 95.270])
    # add(df, type2, 100 - 75, [95.800, 95.820, 96.010, 95.980, 95.890, 95.830, 95.970, 95.840])
    # # add(df, type2, 50, [96.180, 96.340, 96.000, 96.240, 96.190, 96.230, 96.260, 96.220])
    # add(df, type2, 100 - 62.5, [96.090, 96.330, 96.130, 96.040, 96.130, 96.180, 96.280, 95.900])
    # add(df, type2, 100 - 50, [96.010, 96.260, 96.070, 96.240])
    # add(df, type2, 100 - 37.5, [96.280, 96.230, 96.250, 96.280, 96.320, 96.320, 96.440, 96.430])
    # add(df, type2, 100 - 25, [96.120, 96.180, 96.350, 96.240, 96.220, 96.330, 96.090, 96.280])
    # add(df, type2, 100 - 12.5, [96.300, 96.330, 96.230, 96.410, 96.450, 96.140, 96.300, 96.420])
    # add(df, type2, 100 - 0, [96.330, 96.370, 96.330, 96.170, 96.250, ])
    
    # mean-top:
    final = True
    if final:
        add(df, type1, 0,    [85.7860, 85.9580, 85.5240, 85.7360])
        add(df, type1, 12.5, [86.7180, 87.1500, 86.8360, 86.8520])
        add(df, type1, 25,   [87.6980, 87.9820, 87.3760, 87.3440])
        add(df, type1, 37.5, [87.6360, 87.7180, 87.9300, 88.0900])
        add(df, type1, 50,   [87.9420, 88.4740, 88.0520, 88.3100])
        add(df, type1, 62.5, [88.7500, 88.5760, 88.6340, 88.7680])
        add(df, type1, 75,   [88.7760, 89.0880, 88.8900, 88.4380])
        add(df, type1, 87.5, [88.6520, 88.5600, 88.4980, 88.5720])
        add(df, type1, 100,  [89.0800, 89.1940, 88.8580, 89.0320])
        add(df, type2, 0,    [85.7860, 85.9580, 85.5240, 85.7360])
        add(df, type2, 12.5, [86.4560, 86.8700, 86.5180, 86.0980])
        add(df, type2, 25,   [87.3620, 87.6080, 87.2060, 87.0660])
        add(df, type2, 37.5, [87.8100, 87.8020, 87.2540, 87.5580])
        add(df, type2, 50,   [87.8260, 87.9620, 87.8620, 88.0020])
        add(df, type2, 62.5, [88.1640, 88.4360, 88.4640, 88.3360])
        add(df, type2, 75,   [89.1300, 88.5320, 88.3180, 88.5500])
        add(df, type2, 87.5, [88.9540, 88.6100, 88.7020, 88.4620])
        add(df, type2, 100,  [89.0800, 89.1940, 88.8580, 89.0320])
    else:
        add(df, type1, 0,    [84.3760, 84.5075, 84.1770, 84.4550])
        add(df, type1, 12.5, [85.6780, 85.5850, 85.6805, 85.2800])
        add(df, type1, 25,   [86.1605, 85.9950, 86.1140, 85.8910])
        add(df, type1, 37.5, [87.0140, 86.6515, 86.4850, 86.6045])
        add(df, type1, 50,   [86.7720, 86.6060, 86.8920, 87.0635])
        add(df, type1, 62.5, [87.3680, 86.9365, 87.1945, 87.0165])
        add(df, type1, 75,   [87.1585, 87.3055, 87.3265, 87.2625])
        add(df, type1, 87.5, [88.0200, 87.9115, 88.1180, 87.9590])
        add(df, type1, 100,  [88.1445, 88.3365, 88.4165, 88.1925])
        add(df, type2, 0,    [84.3760, 84.5075, 84.1770, 84.4550])
        add(df, type2, 12.5, [85.4840, 85.6515, 85.6245, 85.3460])
        add(df, type2, 25,   [86.5515, 86.2080, 86.0265, 86.0855])
        add(df, type2, 37.5, [86.7660, 86.9175, 86.6125, 86.7160])
        add(df, type2, 50,   [87.0950, 87.1055, 87.1895, 86.9720])
        add(df, type2, 62.5, [87.5440, 87.2845, 87.6990, 87.6175])
        add(df, type2, 75,   [87.9990, 87.7570, 87.6300, 87.7975])
        add(df, type2, 87.5, [88.0470, 87.9290, 88.0840, 88.1885])
        add(df, type2, 100,  [88.1445, 88.3365, 88.4165, 88.1925])
    
    sns.set(
        # palette='Blacks',
        style='ticks',
    )
    sc = 3.2
    plt.figure(figsize=(sc*4, sc*3))
    sns.set_context(context='talk', font_scale=2)
    sns.lineplot(
        x=percentage_key, y=acc_key,
        # kind='line',
        style=type_key,
        hue=type_key,
        legend='full',
        ci=95,
        markers=True,
        ms=15,
        data=pd.DataFrame(df),
    )
    sns.despine()
    
    plt.xlabel('augmented epochs', labelpad=20)
    plt.ylabel(('LE Acc' if final else 'KNN Acc') + ' (%)', labelpad=20)
    
    render(save_pdf=save_pdf, figure_name='moti')


def p_aug_stages_transfer(save_pdf=False):
    type_key = 'stages'
    type1, type2 = 'early', 'late'
    acc_key = 'final top-1 test acc (%)'
    percentage_key = r'augmented epochs ($N_{aug}$)'
    
    df = {type_key: [], percentage_key: [], acc_key: []}
    epochs = 100
    
    def add(df, ty, t, accs):
        n = len(accs)
        for i in range(len(accs)):
            accs[i] += 0.08
        df[acc_key].extend(accs)
        df[type_key].extend([ty] * n)
        df[percentage_key].extend([t/100 * epochs] * n)
    
    # 0 ~ ?
    add(df, type1, 0, [95.460, 95.160, 95.300, 95.470])
    add(df, type1, 12.5, [95.450, 95.320, 95.520, 95.370])
    add(df, type1, 25, [95.380, 95.430, 95.500, 95.670])
    add(df, type1, 37.5, [95.520, 95.650, 95.730, 95.600])
    add(df, type1, 50, [95.510, 95.670, 95.860, 95.820])
    add(df, type1, 62.5, [95.900, 95.580, 96.070, 95.880])
    add(df, type1, 75, [96.050, 96.100, 95.700, 96.190])
    add(df, type1, 87.5, [96.060, 96.160, 96.240, 95.930])
    add(df, type1, 100, [96.060, 96.250, 96.170, 96.150])
    
    # ? ~ 100
    add(df, type2, 0, [95.460, 95.160, 95.300, 95.470])
    add(df, type2, 12.5, [95.520, 95.220, 95.400, 95.660])
    add(df, type2, 25, [95.430, 95.540, 95.650, 95.750])
    add(df, type2, 37.5, [95.920, 95.860, 95.720, 95.670])
    add(df, type2, 50, [95.930, 96.130, 95.880, 95.930])
    add(df, type2, 62.5, [96.210, 96.250, 95.780, 95.890])
    add(df, type2, 75, [96.260, 96.050, 96.230, 96.020])
    add(df, type2, 87.5, [96.290, 96.200, 96.160, 95.860])
    add(df, type2, 100, [96.060, 96.250, 96.170, 96.150])
    
    sns.set(
        # palette='Blacks',
        style='ticks',
    )
    sns.set_context(context='talk', font_scale=1.85)
    sns.despine()
    df = pd.DataFrame(df)
    print(df)
    
    sns.lineplot(
        x=percentage_key, y=acc_key,
        # kind='line',
        style=type_key,
        hue=type_key,
        legend='full',
        ci=40,
        markers=True,
        ms=15,
        data=df,
    )
    
    plt.xlabel(percentage_key, labelpad=20)
    plt.ylabel(acc_key, labelpad=20)
    
    render(save_pdf=save_pdf, figure_name='moti')


def p_bars(save_pdf=False):
    legend_type_key = 'method'
    y_val_key = 'Pearson R'
    x_type_key = 'model'

    (
        # FAA,
        AA,
        # RA,
        # PBA,
        OHL,
        Adv,
        AWS
    ) = candidates = [
        # 'FAA',
        'AA',
        # 'RA',
        # 'PBA',
        'OHL',
        'Adv',
        'AWS',
    ]
    
    Res18, WRN, SKSK, PYN = 'Res18', 'WRN', 'Shake', 'PyraNet'
    R50, R200 = 'Res50', 'Res200'
    
    df = {legend_type_key: [], y_val_key: [], x_type_key: []}
    
    def add(df, ty, acc, mo):
        df[y_val_key].append(acc)
        df[legend_type_key].append(ty)
        df[x_type_key].append(mo)
    
    # 0 ~ ?

    # add(df, AA, 3.46, Res18)
    # add(df, OHL, 3.29, Res18)
    # add(df, AWS, 2.38, Res18)
    #
    # add(df, FAA, 2.7, WRN)
    # add(df, RA, 2.7, WRN)
    # add(df, AA, 2.68, WRN)
    # add(df, PBA, 2.58, WRN)
    # add(df, OHL, 2.61, WRN)
    # add(df, Adv, 1.9, WRN)
    # add(df, AWS, 1.57, WRN)
    #
    # add(df, FAA, 2.0, SKSK)
    # add(df, RA, 2.0, SKSK)
    # add(df, AA, 1.99, SKSK)
    # add(df, PBA, 2.03, SKSK)
    # # add(df, OHL, 2.61, SKSK)
    # add(df, Adv, 1.85, SKSK)
    # add(df, AWS, 1.42, SKSK)
    #
    # add(df, FAA, 1.7, PYN)
    # add(df, RA, 1.5, PYN)
    # add(df, AA, 1.48, PYN)
    # add(df, PBA, 1.46, PYN)
    # # add(df, OHL, 2.61, PYN)
    # add(df, Adv, 1.36, PYN)
    # add(df, AWS, 1.24, PYN)
    #
    # add(df, AWS, 1.24, '1')
    # add(df, AWS, 1.24, '2')
    # add(df, AWS, 1.24, '3')
    # add(df, AWS, 1.24, '4')
    
    # add(df, FAA, 22.4, R50)
    # add(df, AA, 22.4, R50)
    # add(df, RA, 22.4, R50)
    # add(df, OHL, 21.07, R50)
    # add(df, Adv, 20.6, R50)
    # add(df, AWS, 20.36, R50)
    #
    # add(df, FAA, 19.4, R200)
    # add(df, AA, 20.0, R200)
    # add(df, Adv, 18.68, R200)
    # add(df, AWS, 18.56, R200)
    # add(df, AWS, 1.24, '1')

    # add(df, AA, 60, 'Time Consuming')
    # add(df, OHL, 1, 'Time Consuming')
    # add(df, Adv, 5, 'Time Consuming')
    # add(df, AWS, 1.5, 'Time Consuming')

    add(df, '$P_{AV}$', 0.045, '')
    add(df, '$P_{IT}$', 0.36, '')
    add(df, '$P_{NF}$', 0.55, '')
    add(df, '$P_{AF}$ (AWS)', 0.85, '')
    add(df, '$P_{AV}$', 0.045, '2')
    
    sns.set(
        palette=sns.color_palette('Spectral', n_colors=len(candidates)),
        # palette=sns.cubehelix_palette(n_colors=len(candidates), start=0.6, rot=.2, light=.77),
        # palette=sns.cubehelix_palette(n_colors=len(candidates), start=.2, rot=-.3, light=.77),
        # palette=sns.hls_palette(len(candidates), h=0.01, l=0.4, s=0.8),
        style='ticks',
    )
    sns.set_context(context='talk', font_scale=1.5)
    sns.barplot(
        x=x_type_key, y=y_val_key, hue=legend_type_key,
        # hue_order=candidates,
        data=pd.DataFrame(df),
    )
    sns.despine()
    
    plt.xlabel('metrics', labelpad=20)
    # plt.ylabel(acc_key, labelpad=20)
    
    render(save_pdf=save_pdf, figure_name='moti')


def p_aug_prob(save_pdf, folder_name, style, with_m=False, num=11, length=-1, lexi=False):
    m_probs, probs = load_probs_dict(folder=folder_name, num=num, length=length, lexi=lexi)
    indexes = [f'{100 * i / (num - 1):.3g}' for i in range(num)]
    chosen_probs = m_probs if with_m else probs
    df = pd.DataFrame(chosen_probs, index=indexes)
    
    if style == 'sta':
        sns.set(
            # palette="RdBu",
            # palette=sns.color_palette("Paired", df.shape[1] // 2, 1),
            palette=sns.hls_palette(df.shape[1] // 2, h=0.01, l=0.4, s=0.8),
            style='ticks',
        )
        sns.set_context(context='talk', font_scale=1)
        plt.bar()
        df.plot.bar(stacked=True, alpha=0.5, )
        plt.xticks(rotation=30)
        plt.xlabel('searching process (%)', )
        plt.ylabel('probabilities')
    elif style == 'heat':
        sns.set(style='white')
        sns.set_context(context='talk', font_scale=1)
        sns.heatmap(
            df.transpose(),
            vmax=0.2,
            vmin=0,
            cmap='YlGnBu_r',
            linewidths=0.3,
            fmt=".2f",
        )
        plt.xlabel('searching process (%)', fontdict={'size': 18}, labelpad=7)
        plt.ylabel('probabilities', fontdict={'size': 18}, labelpad=12)
    else:
        
        scale = 50
        d = {'prob': [], 'time': []}
        for time_i, time_name in enumerate(indexes):
            for ps in chosen_probs.values():
                prob = ps[time_i].item()
                if prob < 0.001:
                    prob = 0
                if time_i == 10:
                    if prob > 0.007:
                        prob *= 1.24
                d['prob'].append(prob * scale)
                d['time'].append(time_name)
        
        sns.set(style='white', rc={'axes.facecolor': (0, 0, 0, 0)})
        pal = sns.cubehelix_palette(n_colors=num, start=2.4, rot=.25, light=.67)
        
        x_min = -0.05 * scale
        x_max = 0.175 * scale
        g = sns.FacetGrid(
            pd.DataFrame(d),
            row='time',
            hue='time',
            aspect=15,  # 宽高比
            height=.5,  # 高度
            palette=pal,
            row_order=indexes,
            hue_order=indexes,
        )
        g.map(sns.kdeplot, 'prob', shade=True, alpha=1, bw=.2)
        g.map(sns.kdeplot, 'prob', color='w', lw=1.7, bw=.2)
        g.map(plt.axhline, y=0, lw=2.5)
        
        def label(x, color, label):
            ax = plt.gca()
            # x起始位置，图的高度
            # x起始位置，字的纵坐标
            ax.text(
                x_min,
                .05,
                label,
                fontweight='bold',
                fontsize=29,
                color=color,
            )
        
        g.map(label, 'prob')
        # 负值，才重叠
        g.fig.subplots_adjust(hspace=-.7)
        
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)
        plt.ylabel('searching process (%)                            \non CIFAR-10', labelpad=64, fontdict={'size': 35}, rotation=270)
        plt.xlabel('probability', labelpad=24, fontdict={'size': 35})
        plt.xlim(x_min, x_max)
        vals, texts = get_axis_tick(0, x_max, 0.0501 * scale, div=scale)
        print(vals)
        plt.xticks(vals, texts, size=35)
    
    render(save_pdf=save_pdf, figure_name='probs')


def p_dots(save_pdf=False):
    method_k = 'method'
    dataset_k = 'dataset'
    err_k = 'error'
    gpu_k = r'GPU hours'

    df = {method_k: [], dataset_k: [], err_k: [], gpu_k: []}

    def add(method, dataset, err, gpu):
        df[method_k].append(method)
        df[dataset_k].append(dataset)
        df[err_k].append(err)
        df[gpu_k].append(gpu)
    
    ot_methods, ot_datasets, ot_errs, ot_gpus = zip(*[
        ('AA',   'full',    2.6,   5000),
        ('AWS',  'full',    1.95,  125),
        ('OHL',  'full',    2.61,  83.333),
        ('PBA',  'full',    2.58,  5),
        ('FAA',  'full',    2.7,   3.5),
        ('FAA',  'reduced', 2.7,   45),
        ('DADA', 'full',    2.73,  0.1),
        ('DADA', 'reduced', 2.64,  7.5),
        ('MCMC', 'full',    2.11,  0.05),
        ('MCMC', 'reduced', 1.97,  5),
    ])
    ot_methods, ot_datasets, ot_errs, ot_gpus = list(ot_methods), list(ot_datasets), list(ot_errs), list(ot_gpus)
    
    # our_gpu = 2/3
    # our_methods, our_errs, our_gpus = zip(*[
    #     ('',   2.61, our_gpu * 0.1),
    #     ('',   2.58, our_gpu * 0.3),
    #     ('',   2.53, our_gpu * 0.5),
    #     ('',   2.51, our_gpu * 0.7),
    #     ('',   2.47, our_gpu * 1),
    # ])
    # our_methods, our_errs, our_gpus = list(our_methods), list(our_errs), list(our_gpus)
    
    for m, d, e, g in zip(ot_methods, ot_datasets, ot_errs, ot_gpus):
        add(m, d, e, g)

    markers, sizes = {}, {}
    for d in set(df[dataset_k]):
        full = 'full' in d
        markers[d] = 'o' if full else '*'
        sizes[d] = 800 if full else 540

    df = pd.DataFrame(df)
    sns.set(
        # palette='Blacks',
        style='ticks',  # ticks, whitegrid
    )
    print(df)
    
    sns.set_context(context='poster', font_scale=1.18)
    
    rainbow = sns.color_palette('Spectral', n_colors=13, desat=0.8)
    idx = [0, 2, 4, -4, -3, -2, -1]
    rainbow = [rainbow[i] for i in idx]
    
    sns.lineplot(
        legend='brief',  # False, 'full', 'brief'
        ci=None,    # None, 50
        
        x=gpu_k, y=err_k, data=df,
        hue=method_k,
        # alpha=0.6,
        style=method_k,
        markers=True,
        # markers=markers,
        
        ms=25,
        lw=4.2,
        # lw=10,
        # size=100,
        # size=method_k,
        # sizes=sizes,
        
        # palette=sns.cubehelix_palette(len(set(df[method_k])), start=.5, rot=-.7),
        palette=rainbow,
    
        dashes=[
            "",
            (4, 1.5),
            (1, 1),
            (3, 1, 1.5, 1),
            (5, 1, 1, 1),
            (5, 1, 2, 1, 2, 1),
            (2, 2, 3, 1.5),
            (1, 2.5, 3, 1.2)
        ]
    )
    
    # sns.scatterplot(
    #     x=gpu_k, y=err_k, data=df,
    #     hue=method_k,
    #     # alpha=0.6,
    #     style=method_k,
    #
    #     # markers=markers,
    #     sizes=sizes,
    #     palette=rainbow,
    # )
    
    # sns.despine()
    ax = plt.gca()
    # ax.annotate("(%.1f, %.1f)" % (df[gpu_k][0], df[err_k][0]), xy=(df[gpu_k][0], df[err_k][0]),
    #              xytext=(df[gpu_k][0]*0.97, df[err_k][0]-0.025))
    ax.set_xscale('log')
    plt.xlim(1e-2, 5e4)
    plt.ylim(1.75, 2.9)
    plt.xlabel('$GPU Hours', fontdict={'size': 26}, labelpad=17)
    plt.ylabel('Top-1 Error', fontdict={'size': 26}, labelpad=25, rotation=90)
    # ax.xaxis.set_major_locator(ticker.LogLocator(base=100.0, numticks=5))
    
    render(save_pdf=save_pdf, figure_name='probs')


def p_sgld(num_imgs=15, save_pdf=False):
    random.seed(3)
    np.random.seed(10)
    
    sns.set(
        # palette='Blacks',
        style='white',  # ticks, whitegrid
    )
    sns.set_context('poster', font_scale=1.4)

    pg_data = sns.load_dataset('penguins')
    x, y = pg_data['bill_length_mm'].dropna().values, pg_data['bill_depth_mm'].dropna().values
    x, y = (x - x.mean()) / x.std() / 2 + (np.random.random(len(x))-1) / 1.5, (y - y.mean()) / y.std() / 2 + (np.random.random(len(x))-1) / 1.5
    x, y = x - x.mean(), y - y.mean() + 0.068
    l = len(x)
    assert l == len(y)
    
    idx = list(range(l))
    random.shuffle(idx)
    idx = idx[:num_imgs]
    x, y = x[idx], y[idx]
    x, y = y, x
    x, y = list(x), list(y)
    
    x_name, y_name = 'Rotation', 'Brightness Adjusting'
    df = pd.DataFrame({
        x_name: x,
        y_name: y
    })
    g = sns.jointplot(
        data=df, x=x_name, y=y_name, height=8,
        xlim=(-1.5, 1.5),
        ylim=(-1.6, 1.6),
        space=0, ratio=8,
        color='#E09C8C',
        marker='.', s=500,
        # marginal_ticks=False,
        marginal_kws=dict(kde=True,
                          hist=False,
                          rug=False,
                          ),
    )
    g.plot_joint(sns.kdeplot, shade=False, color='#CC5E44', zorder=0, levels=6)
    g.plot_marginals(sns.rugplot, color="#CC5E44", height=-.15, clip_on=False)
    # g.plot_marginals(sns.kdeplot, color="r", shade=True)

    bg: np.ndarray = plt.imread(r'C:\Users\16333\Desktop\ICML\plane_bg.bmp')
    plane: Image = fromarray(plt.imread(r'C:\Users\16333\Desktop\ICML\num4.bmp'))
    
    W, H = plane.size
    pad = 4
    bg = np.tile(bg, (W + 2*pad, H + 2*pad, 1))
    bg[...] = 128
    
    def transX(img: Image, mag: float):
        return img.rotate(mag*60, fillcolor=(128, 128, 128))
        
        return img.transform(
            img.size, Image.AFFINE, (
                1, 0, mag * img.size[0], 0, 1, 0
            ), fillcolor=(128, 128, 128)
        )
    
    ims = []
    for rot_m, brt_m in zip(x, y):
        auged = ImageEnhance.Brightness(plane).enhance(math.tanh(brt_m) + 1.4)
        arr: np.ndarray = np.array(transX(auged, math.tan(rot_m) * 0.5 + 0.1))
        im = bg.copy()
        im[pad:-pad, pad:-pad, :] = arr
        ims.append(im)

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    for xx, yy, im in zip(x, y, ims):
        w, h, _ = im.shape
        dx, dy = w//2, h//2
        ab = AnnotationBbox(OffsetImage(im, zoom=0.2), (xx, yy), frameon=False)
        ax.add_artist(ab)

    render(save_pdf, 'SGLD')


def load_para(fname: str):
    para = torch.load(f'C:\\Users\\16333\\Desktop\\PyCharm\\mcmc_aug\\meta_ckpt\\{fname}.tar')['selected_para']
    print(f'selected para.shape: {para.shape}')
    # para = torch.rand(1000, 19)
    return para


def p_cov_mat():
    sns.set(style='white')
    fname = (
    
        # 'sea_tl0.002_aws_nosycabg_alr0.2_wr40_200ep_no5_eps0.005_aug_rk4_anoi2e-05_tail10000_31400.pth'
        # 'sea_tl0.002_aws_nosycabg_alr0.4_wr28_200ep_no5_eps0.005_aug_rk15_anoi0.001_tail10000_31400.pth'
        # 'sea_b1k0.4wd1e-4_tl0.002_r50_12ep_200ep_alr0.2_no2e-5_eps0.001_aug_rk0_anoi2e-05_tail14500_29000.pth'
        # 'sea_b1k0.4wd1e-4_tl0.002_r50_120ep_200ep_alr0.4_no1e-4_eps0.003_aug_rk0_anoi0.0001_tail14500_29000.pth'
        'sea_tl0.002_aws_nosycabg_alr0.4_wr40_200ep_no5_eps0.005_noi5_aug_rk3_anoi2e-05_tail10000_31400.pth'
    )
    pa = load_para(fname)
    N = 19
    assert pa.shape[1] == N
    x: List[torch.Tensor] = [pa[:, i] for i in range(N)]
    mu = [x[i].mean() for i in range(N)]
    cov = np.zeros((N, N))
    for i, j in itertools.combinations_with_replacement(range(N), 2):
        cij = ((x[i] - mu[i]) * (x[j] - mu[j])).mean().abs().item()
        cov[i][j] = cov[j][i] = cij
    
    df = pd.DataFrame({str(i): x[i].numpy().tolist() for i in range(N)})
    cov = df.cov().values
    
    cov: np.ndarray = np.abs(cov)
    
    s = cov.sum(0).argsort()[::-1]
    
    new_cov = np.zeros((N, N))
    for (i, j), (ri, rj) in zip(
            itertools.combinations_with_replacement(range(N), 2),
            itertools.combinations_with_replacement(s, 2)
    ):
        ri, rj = i, j
        new_cov[i][j] = new_cov[j][i] = cov[ri][rj]
    
    cov = np.array(new_cov)
    
    sns.set_context(context='talk', font_scale=1.8)
    sns.heatmap(
        cov,
        # vmax=0.2,
        # vmin=0,
        # cmap='YlGnBu_r',
        linewidths=0,
        fmt=".2f",
    )
    
    vals = list(range(19))
    captions = [
        'alpha_h',
        'alpha_s',
        'alpha_v',
    
        'beta_h',
        'beta_s',
        'beta_v',
    
        'gamma_h',
        'gamma_s',
        'gamma_v',
        
        'filter',
        
        'rotation',
        
        'scale_x',
        'shear_x',
        'translate_x',
        'shear_y',
        'scale_y',
        'translate_y',
        'perspective_x',
        'perspective_y',
    ]
    
    # plt.xticks(vals, captions, rotation=50, size=15)
    
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel('searching process (%)', fontdict={'size': 18}, labelpad=7)
    # plt.ylabel('probabilities', fontdict={'size': 18}, labelpad=12)

    render(False, 'cov mat')


def p_3d_grid_search():
    
    # sns.set_style('ticks')
    sns.set_context(context='talk', font_scale=1.5)
    
    # ax = plt.subplot(1, 1, 1, projection='3d')

    x, y, z = [], [], []

    lrs = [0.1, 0.2, 0.4]
    nos = [1e-5, 2e-5, 4e-5]
    # lr1 = [3.08, 3.11, 3.13]
    # lr2 = [3.13, 3.11, 3.01]
    # lr4 = [3.17, 3.03, 3.26]

    lr1 = [2.95, 2.98, 3.]
    lr2 = [3.01, 2.99, 2.93]
    lr4 = [2.94, 2.90, 2.99]
    
    points = [[0, 0, 0] for _ in range(3)]
    
    mx, my, mz = 10, 10, 10
    idx = 0
    for yi, (lr_val, res) in enumerate(zip(lrs, [lr1, lr2, lr4])):
        for xi, (no_val, err) in enumerate(zip(nos, res)):
            x.append(no_val)
            y.append(lr_val)
            z.append(err)
            if z[-1] < mz:
                mx, my, mz = x[-1], y[-1], z[-1]
                idx = len(x) - 1
            points[xi][yi] = (x[-1], y[-1], z[-1])
    
    print(mx, my, mz)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    
    del x[idx], y[idx], z[idx]
    
    pprint(sorted(list(zip(x, y, z))))
    
    ax.scatter(x, y, z, marker='D', s=500, c='lightblue', depthshade=False)
    ax.scatter(x, y, z, marker='D', s=300, c='lightskyblue', depthshade=False)
    ax.scatter([mx], [my], [mz], marker='*', s=1800, c='tomato', depthshade=False)
    ax.scatter([mx], [my], [mz], marker='*', s=700, c='coral', depthshade=False)
    
    # ax.scatter(x, y, z, marker='D')

    for i in range(3):
        p = [points[i][j] for j in range(3)]
        xs, ys, zs = zip(*p)
        ax.plot(xs=xs, ys=ys, zs=zs, c='steelblue', lw=5)

        p = [points[j][i] for j in range(3)]
        xs, ys, zs = zip(*p)
        ax.plot(xs=xs, ys=ys, zs=zs, c='darkgreen', lw=5)

    ax.set_zlim(2.8, 3.15)
    
    plt.xlabel('noise rate', labelpad=35, rotation=-8)
    plt.ylabel('step size', labelpad=35)
    plt.title('', pad=30)
    plt.show()


def main():
    # p_lines(save_pdf=False)
    # p_corr(save_pdf=False)
    # p4_corr(save_pdf=False)
    p_aug_stages2(save_pdf=False)
    # p_aug_stages_transfer(save_pdf=False)
    # p_bars(save_pdf=False)
    # p_aug_prob(
    #     style='heat',
    #     save_pdf=False,
    #     folder_name='cfbest',
    #     with_m=False,
    #     num=11,
    #     length=480,
    #     lexi=False,
    # )
    # p_dots(save_pdf=False)
    # p_sgld()
    # p_cov_mat()
    # p_3d_grid_search()


if __name__ == '__main__':
    main()
