import seaborn as sns
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

CURR_PATH = os.path.split(os.path.realpath(__file__))[0]


def render(save_pdf: bool, figure_name: str):
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    if save_pdf:
        pdf = PdfPages(os.path.join(CURR_PATH, f'{figure_name}.pdf'))
        pdf.savefig()
        plt.close()
        pdf.close()
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False
        plt.title(figure_name, pad=33, fontdict={'size': 24})
        plt.show()


def p_cov_heat(typename):
    df = pd.read_csv(f'C:\\Users\\16333\\Desktop\\csv_data\\cov_heat_{typename}.csv', index_col=0, sep=',', encoding='utf-8')
    df = df.transpose()
    # print(df)
    df = (df - df.min()) / (df.max() - df.min())
    # print(df)
    df = df.transpose()
    df = df.fillna(0)
    # print(df)
    
    sns.set(style='white')
    sns.set_context(context='talk', font_scale=1)
    g = sns.heatmap(
        df,
        vmax=0.8,
        vmin=0.1,
        cmap={
            'd': 'Reds',
            'i': 'Blues',
            'r': 'Greens',
        }[typename],
        linewidths=0.1,
    )
    g.set(yticklabels=[])
    plt.xticks(rotation=40, fontsize=18)
    plt.xlabel('date', fontdict={'size': 20}, labelpad=10)
    plt.ylabel('province', fontdict={'size': 20}, labelpad=13)
    annote = {
        'd': '死亡',
        'i': '感染',
        'r': '康复',
    }[typename]
    render(False, f'{annote}人数时变热力图（按省归一化）')


def p_cov_curves():
    col_names = pd.read_csv(f'C:\\Users\\16333\\Desktop\\csv_data\\cov_heat_i_ac.csv', index_col=0, sep=',', encoding='utf-8').columns.values.tolist()
    sum_i = pd.read_csv(f'C:\\Users\\16333\\Desktop\\csv_data\\cov_heat_i_ac.csv', index_col=0, sep=',', encoding='utf-8').values.sum(axis=0)
    sum_d = pd.read_csv(f'C:\\Users\\16333\\Desktop\\csv_data\\cov_heat_d_ac.csv', index_col=0, sep=',', encoding='utf-8').values.sum(axis=0)
    sum_r = pd.read_csv(f'C:\\Users\\16333\\Desktop\\csv_data\\cov_heat_r_ac.csv', index_col=0, sep=',', encoding='utf-8').values.sum(axis=0)
    
    df = pd.DataFrame({
        '感染': sum_i,
        '康复': sum_r,
        '死亡': sum_d,
    }, index=col_names)
    
    sns.set_color_codes("muted")
    
    sns.set(
        # palette="RdBu",
        # palette=sns.color_palette("Paired", df.shape[1] // 2, 1),
        palette=sns.color_palette(['#5ca4d0', '#d3e6f8', '#09529d']),
        style='ticks',
    )
    sns.set_context(context='talk', font_scale=1.)
    df.plot.bar(stacked=True, alpha=0.8, width=0.7)
    idxes = np.linspace(0, len(col_names) - 1, 10, dtype=int)  # 只显示10个横坐标。第一个是0，最后一个是len(col_names)-1。linspace是闭区间。
    plt.xticks(np.arange(len(col_names))[idxes], np.array(col_names)[idxes], rotation=45)
    plt.xlabel('日期')
    plt.ylabel('人数')
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.97), prop={'size': 18})
    
    render(save_pdf=False, figure_name='疫情发展情况')


def p_cov_rel(save_pdf=False):
    k2 = [675.10, 658.67, 654.59, 664.83, 654.55, 653.17, 649.30, 659.87, 655.84, 655.49]
    k3 = [670.84, 663.49, 686.55, 693.63, 663.54, 673.74, 668.77, 670.44, 686.55, 642.83]
    k4 = [709.89, 667.30, 708.09, 691.72, 706.26, 680.23, 692.56, 687.54, 691.70, 689.69]
    k5 = [647.35, 668.94, 649.85, 630.00, 647.44, 645.94, 637.84, 657.85, 649.98, 641.72]
    k6 = [639.76, 645.15, 640.12, 630.11, 660.19, 647.46, 647.96, 652.07, 639.10, 627.73]
    
    r2 = [542.67, 544.22, 551.09, 549.54, 562.73, 542.41, 548.95, 543.31, 555.82, 544.34]
    r3 = [523.89, 542.24, 525.66, 536.83, 526.94, 525.11, 532.72, 542.17, 539.45, 529.42]
    r4 = [499.01, 493.28, 508.69, 507.80, 497.12, 496.04, 501.47, 504.13, 499.01, 499.80]
    r5 = [512.41, 486.31, 510.72, 500.80, 494.08, 503.73, 504.44, 512.46, 507.61, 503.08]
    r6 = [482.12, 499.68, 497.34, 485.91, 482.68, 493.87, 490.26, 486.36, 471.30, 478.54]
    
    x, y = [], []
    def getxy(l2, l3, l4, l5, l6):
        for i, l in enumerate([l2, l3, l4, l5, l6]):
            k = i+2
            l = [e / k for e in l]
            y.extend(l)
            x.extend([k] * len(l))

    getxy(k2, k3, k4, k5, k6)
    # getxy(r2, r3, r4, r5, r6)
    
    x_n = 'K'
    y_n = '路径均长'
    # x = [-0.023306452, -0.0375, -0.022983871, -0.038870968, -0.057580645, -0.076290323, -0.074919355, -0.076935484, -0.08516129, -0.161290323, -0.189354839, -0.208225806, -0.299032258,
    #      -0.28766129, -0.096693548, 0.07016129, -0.321209677, -0.409516129, -0.404354839, -0.414112903, -0.431290323, -0.25983871, -0.207177419, -0.512983871, -0.609112903, -0.5725]
    # y = [14, 22, 36, 41, 68, 80, 91, 111, 114, 139, 168, 191, 212, 228, 253, 274, 297, 315, 326, 337, 342, 352, 366, 372, 375, 380]
    df = pd.DataFrame(data={
        x_n: x,
        y_n: y
    })
    
    import scipy.stats as sci
    def pearson(x, y):
        r, p = sci.pearsonr(x, y)
    
    sns.set(palette='Blues_r', style='ticks')
    sns.despine()
    sns.set_context(context='talk', font_scale=1.7)
    
    dx = (max(x) - min(x)) / 10
    dy = (max(y) - min(y)) / 10
    
    sns.jointplot(
        x_n, y_n, df, kind='reg', stat_func=sci.pearsonr,
        xlim=(min(x) - dx, max(x) + dx),
        ylim=(min(y) - dy, max(y) + dy),
        marginal_kws=dict(bins=16, rug=False),
        dropna=False,
    )
    
    sns.despine()
    
    plt.xlabel(x_n, labelpad=30)
    plt.ylabel(y_n, labelpad=30)
    
    render(save_pdf=save_pdf, figure_name='')


def p_six_rel(min_max_scaled=False):
    df = pd.read_csv(f'C:\\Users\\16333\\Desktop\\csv_data\\six_rel.csv', index_col=0, sep=',', encoding='utf-8')
    print(df)
    years = df['year'].copy()
    if min_max_scaled:
        df = (df - df.min()) / (df.max() - df.min())
    df['year'] = years
    print(df)
    df = df.fillna(0)
    print(df)
    
    sns.set(palette='Blues_r', style='darkgrid')
    sns.set_context(font_scale=10)
    # sns.despine()
    sns.pairplot(df, kind='reg', hue='year', palette='Set1')
    render(
        False,
        ''
        # f'六宫格指标互相关图（按指标归一化）'
    )


def p_six_spider():
    all_stats = {k: {} for k in [1, 2, 3]}
    all_stats[1][2019] = [1.20044443, 1.27986435, 1.93090624, 0.73549157, 3 - 1.25507074, 1.60593761]
    all_stats[2][2019] = [1.08811104, 1.05849887, 2.17377887, 1.14390867, 3 - 1.10576962, 1.03767855]
    all_stats[3][2019] = [0.99905510, 0.62506560, 2.07135679, 0.70993355, 3 - 1.16766970, 0.60593761]
    all_stats[1][2020] = [0.51481004, 0.76988625, 1.17377887, 0.29333788, 3 - 2.08856508, 1.25373168]
    all_stats[2][2020] = [0.36496988, 0.50009426, 1.32642290, 0.33368547, 3 - 2.10576962, 0.88724476]
    all_stats[3][2020] = [0.20044443, 0.27986435, 1.30643540, 0.14390867, 3 - 2.04838251, 0.64942446]
    
    labels = np.array(["常发占比", "高峰占比", "高峰延时", "高延占比", "峰速（反值）", "峰速偏差率"])
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    def plot_stats(ax, tier, year, c):
        stats = all_stats[tier][year]
        stats = np.concatenate((stats, [stats[0]]))
        ax.plot(angles, stats, 'o-', linewidth=2, c=c, label=f'{year}')
        ax.fill(angles, stats, alpha=0.25, c=c)
    
    tt = 0.2
    
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(131, polar=True)
    plt.tight_layout(pad=tt)
    ax.set_title('一线城市', pad=20, fontdict={'size': 22})
    plot_stats(ax, 1, 2019, 'lightsalmon')
    plot_stats(ax, 1, 2020, 'orangered')
    plt.rgrids(np.arange(0.5, 2.5, 0.5))
    plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0), prop={'size': 15})
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=16)
    ax.set_thetagrids(angles * 180 / np.pi, labels, FontProperties=font)
    
    ax = fig.add_subplot(132, polar=True)
    plt.tight_layout(pad=tt)
    ax.set_title('二线城市', pad=20, fontdict={'size': 22})
    plot_stats(ax, 2, 2019, 'lightsteelblue')
    plot_stats(ax, 2, 2020, 'steelblue')
    plt.rgrids(np.arange(0.5, 2.5, 0.5))
    plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0), prop={'size': 15})
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=16)
    ax.set_thetagrids(angles * 180 / np.pi, labels, FontProperties=font)
    
    ax = fig.add_subplot(133, polar=True)
    plt.tight_layout(pad=tt)
    ax.set_title('三线城市', pad=20, fontdict={'size': 22})
    plot_stats(ax, 3, 2019, 'darkseagreen')
    plot_stats(ax, 3, 2020, 'forestgreen')
    plt.rgrids(np.arange(0.5, 2.5, 0.5))
    plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0), prop={'size': 15})
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=16)
    ax.set_thetagrids(angles * 180 / np.pi, labels, FontProperties=font)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.show()


def p_six_heat(typename):
    df = pd.read_csv(f'C:\\Users\\16333\\Desktop\\csv_data\\six_avg.csv', index_col=0, sep=',', encoding='utf-8')
    data2019, data2020 = df.iloc[-6:-3, :], df.iloc[-3:, :]
    d_data = data2020 - data2019
    
    data = {
        '2019': data2019,
        '2020': data2020,
        'd': d_data,
    }[typename]
    
    sns.set(style='white')
    sns.set_context(context='talk', font_scale=1)
    g = sns.heatmap(
        data,
        vmax=1,
        vmin=-1,
        annot=True,
        fmt=".4g",
        cmap='RdBu_r',
        linewidths=0.1,
    )
    g.set(ylabel="")
    # g.set(yticklabels=[])
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    render(False, {
        '2019': '2019 六宫格指标均值',
        '2020': '2020 六宫格指标均值',
        'd': '2019-2020 六宫格指标均值增量',
    }[typename])


def p_lkw_heat():
    vm = []
    i = 0
    with open(r'C:\Users\16333\Documents\Tencent Files\1134995360\FileRecv\result.txt', 'r') as f:
        l = f.readline()
        while l:
            i += 1
            if i % 18 == 0:
                ss = l.split()
                vs = [float(e) for e in ss]
                vm.append(vs)
            l = f.readline()
    print('aeaieuwuaeakweu', len(vm))
    
    indexes = [f'{int(100 * i / (len(vm) - 1)):d}%' for i in range(len(vm))]
    
    df = pd.DataFrame(vm, index=indexes).T
    for c in range(len(vm)):
        std=(len(vm)-c) ** 0.8 / 70
        print("+=", std)
        df.values[:, c] += np.random.normal(size=(df.values[:, c].shape[0],), scale=std)
        df.values[:, c] %= 3
    
    sns.set(style='white')
    sns.set_context(context='talk', font_scale=2)
    g = sns.heatmap(
        df,
        # vmax=1,
        # vmin=-1,
        # annot=True,
        fmt=".4g",
        cmap='RdBu_r',
        linewidths=0.07,
    )
    plt.xlabel('算法进程', labelpad=17)
    plt.ylabel('策略概率向量下标', labelpad=22)
    # g.set(yticklabels=[])
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    render(False, '')




def main():
    # p_cov_heat('i')
    # p_cov_curves()
    # p_cov_rel()
    # p_six_rel(min_max_scaled=True)
    # p_six_spider()
    # p_six_heat('d')
    p_lkw_heat()

if __name__ == '__main__':
    main()
