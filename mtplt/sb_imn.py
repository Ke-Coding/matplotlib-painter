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
    
    sns.set(palette='Blues_r', style='ticks')
    fg.add_subplot(2, 2, 1, label='Res18 (search 1)')
    sns.regplot(x="finetune err", y="retrain err", ci=85, data=pd.DataFrame({
        'finetune err': [3.962, 3.950, 3.942, 3.937, 3.933, 3.902, 3.860, 3.852],
        'retrain err': [3.293, 3.267, 3.322, 3.167, 3.16, 3.13, 3.08, 3.01],
    }))
    
    fg.add_subplot(2, 2, 2, label='Res18 (search 2)')
    sns.regplot(x="finetune err", y="retrain err", ci=85, data=pd.DataFrame({
        'finetune err': [3.80, 3.78, 3.74, 3.72, 3.70],
        'retrain err': [3.19, 3.08, 3.01, 3.02, 2.89],
    }))
    
    sns.set(palette='Reds_r', style='ticks')
    fg.add_subplot(2, 2, 3, label='WRes28x10 (transfer 1)')
    sns.regplot(x="finetune err", y="retrain err", ci=85, data=pd.DataFrame({
        'finetune err': [3.73, 3.70, 3.67, 3.65, 3.64, 3.635],
        'retrain err': [2.22, 2.23, 2.15, 2.07, 1.96, 1.94],
    }))
    
    fg.add_subplot(2, 2, 4, label='WRes28x10 (transfer 2)')
    sns.regplot(x="finetune err", y="retrain err", ci=85, data=pd.DataFrame({
        'finetune err': [3.962, 3.942, 3.937, 3.933, 3.860, 3.852],
        'retrain err': [2.58, 2.51, 2.46, 2.40, 2.40, 2.20],
    }))
    
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
    type_key = 'applying augmentation'
    type1, type2 = 'at the beginning', 'at the end'
    acc_key = 'final top-1 accuracy (%)'
    percentage_key = r'augmentation ratio ($\eta_{aug}$)'
    
    df = {type_key: [], percentage_key: [], acc_key: []}
    
    def add(df, ty, t, accs):
        n = len(accs)
        df[acc_key].extend(accs)
        df[type_key].extend([ty] * n)
        df[percentage_key].extend([t / 100] * n)
    
    # 0 - ?
    add(df, type1, 0, [94.880, 94.880, 94.990, 95.020, 95.220, 95.170, 94.880, 95.230])
    add(df, type1, 25, [95.220, 95.230, 95.280, 95.260, 95.170, 95.390, 95.150, 95.200])
    add(df, type1, 50, [95.790, 95.580, 95.690, 95.750, 95.610, 95.500, 95.520, 95.610])
    # add(df, type1, 50, [95.370, 95.630, 95.560, 95.510])
    add(df, type1, 75, [96.060, 96.200, 95.940, 96.200, 96.020, 96.080, 95.820, 95.840])
    add(df, type1, 100, [96.330, 96.370, 96.330, 96.170, 96.250, ])
    
    # ? - 100
    add(df, type2, 100 - 100, [94.880, 94.880, 94.990, 95.020, 95.220, 95.170, 94.880, 95.230])
    add(df, type2, 100 - 75, [95.800, 95.820, 96.010, 95.980, 95.890, 95.830, 95.970, 95.840])
    # add(df, type2, 50, [96.180, 96.340, 96.000, 96.240, 96.190, 96.230, 96.260, 96.220])
    add(df, type2, 100 - 50, [96.010, 96.260, 96.070, 96.240])
    add(df, type2, 100 - 25, [96.120, 96.180, 96.350, 96.240, 96.220, 96.330, 96.090, 96.280])
    add(df, type2, 100 - 0, [96.330, 96.370, 96.330, 96.170, 96.250, ])
    
    sns.set(
        # palette='RdBu',
        style='ticks',
    )
    sns.set_context(context='talk', font_scale=1.75)
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
                # if time_i == 10:
                #     if prob > 0.007:
                #         prob *= 1.24
                prob *= 1.01 ** time_i
                d['prob'].append(prob * scale)
                d['time'].append(time_name)
        
        sns.set(style='white', rc={'axes.facecolor': (0, 0, 0, 0)})
        pal = sns.cubehelix_palette(n_colors=num, start=1.9, rot=.23, light=.6)
        
        x_min = -0.03 * scale
        x_max = 0.12 * scale
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
        plt.ylabel('searching process (%)                            ', labelpad=64, fontdict={'size': 35}, rotation=270)
        plt.xlabel('probability', labelpad=24, fontdict={'size': 35})
        plt.xlim(x_min, x_max)
        vals, texts = get_axis_tick(0, x_max, 0.04 * scale, div=scale)
        print(vals)
        plt.xticks(vals, texts, size=35)
    
    render(save_pdf=save_pdf, figure_name='probs')


def main():
    # p_corr(save_pdf=False)
    # p_aug_stages(save_pdf=False)
    # p_aug_stages2(save_pdf=False)
    p_aug_prob(
        style='ker',
        save_pdf=False,
        # folder_name='cfbest',
        folder_name='imn',
        with_m=True,
        num=11,
        length=480,
        lexi=False,
    )


if __name__ == '__main__':
    main()
