import numpy as np
import matplotlib.pyplot as plt

data = [[6.638, 1.742, 7.513, 5.728, 3.215],
        [5.823, 3.811, 7.845, 9.938, 1.654],
        [8.913, 8.055, 1.558, 4.981, 6.035],
        [7.841, 8.185, 1.506, 1.463, 6.538],
        [9.361, 3.315, 3.434, 7.813, 5.269]]

columns = ('name1', 'name2', 'name3', 'name4', 'name5')  # 横坐标数值
rows = ['genus %d' % x for x in (1, 2, 3, 4, 5)]  # 表格第一列名称

values = np.arange(0, 40, 10)  # 纵坐标数值

# 选取几种颜色

colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) * 0.3
bar_width = 0.4

y_offset = np.array([0.0] * len(columns))

# 绘制条形图
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset * data[row]
    cell_text.append(['%1.1f' % (x) for x in y_offset])
# 反转颜色和标签
colors = colors[::-1]
cell_text.reverse()

# 在x轴底部添加一个表

the_table = plt.table(cellText=data,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("Expression/10^-5")
plt.yticks(values, ['%d' % val for val in values])
plt.xticks([])
plt.title('Otu Abundance')

plt.show()
