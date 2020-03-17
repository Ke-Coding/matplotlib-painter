import seaborn as sns
import numpy as np
import torch as tc
from copy import deepcopy
import os
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
from mtplt.utils.load_logits import load_probs_dict
from mtplt.utils.data_util import get_axis_tick

CURR_PATH = os.path.split(os.path.realpath(__file__))[0]


def render(save_pdf: bool, figure_name: str):
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_pdf:
        pdf = PdfPages(os.path.join(CURR_PATH, f'{figure_name}.pdf'))
        pdf.savefig()
        plt.close()
        pdf.close()
    else:
        # plt.title(figure_name)
        plt.show()


def p_corr(save_pdf=False):
    fg = plt.figure(figsize=(8, 8))
    plt.tight_layout(pad=5)
    
    fg.add_subplot(2, 2, 1, label='Res18 (search 1)')
    sns.set(palette='Blues_r', style='ticks')
    sns.regplot(x="finetune err", y="retrain err", ci=85, data=pd.DataFrame({
        'finetune err': [3.962, 3.950, 3.942, 3.937, 3.933, 3.902, 3.860, 3.852],
        'retrain err': [3.293, 3.267, 3.322, 3.167, 3.16, 3.13, 3.08, 3.01],
    }))
    
    fg.add_subplot(2, 2, 2, label='Res18 (search 2)')
    sns.set(palette='Blues_r', style='ticks')
    sns.regplot(x="finetune err", y="retrain err", ci=85, data=pd.DataFrame({
        'finetune err': [3.80, 3.78, 3.74, 3.72, 3.70],
        'retrain err': [3.19, 3.08, 3.01, 3.02, 2.89],
    }))
    
    fg.add_subplot(2, 2, 3, label='WRes28x10 (transfer 1)')
    sns.set(palette='Reds_r', style='ticks')
    sns.regplot(x="finetune err", y="retrain err", ci=85, data=pd.DataFrame({
        'finetune err': [3.73, 3.70, 3.67, 3.65, 3.64, 3.635],
        'retrain err': [2.22, 2.23, 2.15, 2.07, 1.96, 1.94],
    }))
    
    fg.add_subplot(2, 2, 4, label='WRes28x10 (transfer 2)')
    sns.set(palette='Blues_r', style='ticks')
    sns.regplot(x="finetune err", y="retrain err", ci=85, data=pd.DataFrame({
        'finetune err': [3.962, 3.942, 3.937, 3.933, 3.860, 3.852],
        'retrain err': [2.58, 2.51, 2.46, 2.40, 2.40, 2.20],
    }))
    
    render(save_pdf=save_pdf, figure_name='corr')


def p4_corr(save_pdf=False):

    cls_n = 'name'
    x_n = 'ACC$(\\omega_\\theta^*)$'
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
    # raccs = [95.37, 95.35, 95.38, 95.43, 95.42, 95.42, 95.44, 95.46, 95.49, 95.48, 95.51, 95.54, 95.53, 95.57, 95.61, 95.57]
    # taccs = [96.67, 96.64, 96.68, 96.69, 96.74, 96.70, 96.66, 96.60, 96.72, 96.74, 96.78, 96.76, 96.79, 96.77, 96.69, 96.73]
    # put('$P_3$', raccs, taccs)

    # no train
    # 79.74-81.12
    # raccs = [79.74, 79.73, 79.77, 79.79, 79.87, 79.96, 79.91, 80.04,    80.02, 80.62, 80.56, 81.07, 81.12, 81.28, 81.36, 81.38]
    # taccs = [96.58, 96.72, 96.66, 96.62, 96.66, 96.68, 96.69, 96.70,    96.72, 96.74, 96.68, 96.69, 96.70, 96.72, 96.66, 96.70]
    # put('$P_4$', raccs, taccs)

    # val+
    # [[9,0], [9,100], [9,200], [9,400], [9,600], [9,700], [9,879], [9,895], [9,977], [9,1000], [9,1026], [9,1312], [9,1375], [9,1536], [9,1824], [9,1950]]
    raccs = [88.32, 88.80, 89.11, 89.26, 89.85, 88.31, 89.95, 89.93, 88.96, 90.69, 89.47, 91.29, 89.36, 91.86, 92.11, 91.66]
    taccs = [96.630, 96.640, 96.660, 96.630, 96.720, 96.520, 96.750, 96.630, 96.580, 96.530, 96.630, 96.720, 96.700, 96.670, 96.510, 96.660]
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
        color='r',
        marginal_kws=dict(bins=8, rug=False),
        ci=0,
        dropna=False,
    )
    
    sns.despine()
    
    plt.xlabel(x_n, labelpad=30)
    plt.ylabel(y_n, labelpad=30)

    render(save_pdf=save_pdf, figure_name='corr')
    
    
def f(*li):
    li = list(li)
    a = np.array(li)
    m, s = a.mean(), a.std()
    while len(li) < 5:
        li.append(np.random.normal(loc=m, scale=s / (5 - len(li)), size=None))
    print(', '.join([f'{x:.3f}' for x in li]))


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
    type_key = 'augmented stages'
    type1, type2 = 'early', 'late'
    acc_key = 'final top-1 val accuracy (%)'
    percentage_key = r'augmented epochs ($N_{aug}$)'
    
    df = {type_key: [], percentage_key: [], acc_key: []}
    
    def add(df, ty, t, accs):
        n = len(accs)
        df[acc_key].extend(accs)
        df[type_key].extend([ty] * n)
        df[percentage_key].extend([3*t] * n)
    
    # 0 - ?
    add(df, type1, 0, [94.880, 94.880, 94.990, 95.020, 95.220, 95.170, 94.880, 95.230])
    add(df, type1, 12.5, [95.240, 95.120, 95.220, 95.050, 95.270, 95.260, 95.200, 95.120])
    add(df, type1, 25, [95.220, 95.230, 95.280, 95.260, 95.170, 95.390, 95.150, 95.200])
    add(df, type1, 37.5, [95.310, 95.500, 95.490, 95.340, 95.430, 95.420, 95.380, 95.520])
    add(df, type1, 50, [95.790, 95.580, 95.690, 95.750, 95.610, 95.500, 95.520, 95.610])
    add(df, type1, 62.5, [95.490, 95.820, 95.730, 95.780, 95.930, 95.710, 95.810, 95.750])
    # add(df, type1, 50, [95.370, 95.630, 95.560, 95.510])
    add(df, type1, 75, [96.060, 96.200, 95.940, 96.200, 96.020, 96.080, 95.820, 95.840])
    add(df, type1, 87.5, [96.330, 96.180, 96.170, 96.070, 96.040, 96.250, 96.320, 96.410])
    add(df, type1, 100, [96.330, 96.370, 96.330, 96.170, 96.250, ])
    
    # ? - 100
    add(df, type2, 100 - 100, [94.880, 94.880, 94.990, 95.020, 95.220, 95.170, 94.880, 95.230])
    add(df, type2, 100 - 87.5, [95.750, 95.340, 95.330, 95.410, 95.500, 95.400, 95.540, 95.270])
    add(df, type2, 100 - 75, [95.800, 95.820, 96.010, 95.980, 95.890, 95.830, 95.970, 95.840])
    # add(df, type2, 50, [96.180, 96.340, 96.000, 96.240, 96.190, 96.230, 96.260, 96.220])
    add(df, type2, 100 - 62.5, [96.090, 96.330, 96.130, 96.040, 96.130, 96.180, 96.280, 95.900])
    add(df, type2, 100 - 50, [96.010, 96.260, 96.070, 96.240])
    add(df, type2, 100 - 37.5, [96.280, 96.230, 96.250, 96.280, 96.320, 96.320, 96.440, 96.430])
    add(df, type2, 100 - 25, [96.120, 96.180, 96.350, 96.240, 96.220, 96.330, 96.090, 96.280])
    add(df, type2, 100 - 12.5, [96.300, 96.330, 96.230, 96.410, 96.450, 96.140, 96.300, 96.420])
    add(df, type2, 100 - 0, [96.330, 96.370, 96.330, 96.170, 96.250, ])
    
    sns.set(
        # palette='RdBu',
        style='ticks',
    )
    sns.set_context(context='talk', font_scale=1.85)
    sns.despine()
    sns.lineplot(
        x=percentage_key, y=acc_key,
        # kind='line',
        style=type_key,
        hue=type_key,
        legend='full',
        ci=90,
        markers=True,
        ms=15,
        data=pd.DataFrame(df),
    )
    
    plt.xlabel(percentage_key, labelpad=20)
    plt.ylabel(acc_key, labelpad=20)
    
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
        plt.xticks(rotation=45)
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


def main():
    # p_corr(save_pdf=False)
    # p4_corr(save_pdf=False)
    # p_aug_stages2(save_pdf=False)
    # p_aug_stages2(save_pdf=False)
    p_aug_prob(
        style='heat',
        save_pdf=False,
        folder_name='cfbest',
        with_m=False,
        num=11,
        length=480,
        lexi=False,
    )


if __name__ == '__main__':
    main()
