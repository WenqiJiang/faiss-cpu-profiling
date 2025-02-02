import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.style.use('ggplot')

def get_default_colors():

  default_colors = []
  for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
      default_colors.append(color["color"])
      # print(color["color"], type(color["color"]))

  return default_colors

default_colors = get_default_colors()

# SIFT1000M, IVF65536, nprobe=51 (R@100=95%)
y_QPS = np.array([4776, 4758, 4672, 4751, 4747, 4720, 4428, 4508])
y_recall = np.array([0.3334, 0.7783, 0.8572, 0.9067, 0.9198, 0.9244, 0.9256, 0.9259]) * 100

x_labels = ['topK=1', 'topK=10', 'topK=20', 'topK=50',\
    'topK=100', 'topK=200', 'topK=500', 'topK=1000']

x = np.arange(len(x_labels))  # the label locations
width = 0.3  # the width of the bars    

# fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 2))

fig, ax0 = plt.subplots(1, 1, figsize=(12, 2))
ax1 = ax0.twinx()

label_font = 12
tick_font = 10
tick_label_font = 9
legend_font = 10
title_font = 14

rects = ax0.bar(x, y_QPS, width, color=default_colors[0])
line, = ax1.plot(x, y_recall, marker='X', markersize=10, color=default_colors[1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Recall (%)', fontsize=label_font)
ax0.set_ylabel('QPS', fontsize=label_font)
ax0.set_xticks(x)
ax0.set_xticklabels(x_labels, fontsize=tick_label_font)
plt.xticks(rotation=0)

ax0.legend([rects, line], ["QPS", "Recall"], facecolor='white', framealpha=1, frameon=True, loc=(0.01, 0.85), fontsize=legend_font, ncol=2)


# ax.set_title('{} R@{}={}: {:.2f}x over CPU, {:.2f}x over GPU'.format(
#     dbname, topK, int(recall_goal*100), best_qps_fpga/best_qps_cpu, best_qps_fpga/best_qps_gpu), 
#     fontsize=label_font)

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            notation = '{:.0f}'.format(height)
        else:   
            notation = 'N/A'
        ax.annotate(notation,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=tick_font, horizontalalignment='center', rotation=90)


# autolabel(rects, ax0)

ax1.set(ylim=[0, 100])
ax0.set(ylim=[0, np.amax(y_QPS) * 1.2])

plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./out_img/cpu_throughput_recall_experiment_5_topK_SIFT1000M.png', transparent=False, dpi=200, bbox_inches="tight")
plt.show()
