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

# SIFT1000M, R@100 recall
y_QPS = np.array([31867, 20938, 19087, 10750, 6743, 3822, 2082, 1101])
y_recall = np.array([0.3465, 0.4892, 0.6370, 0.7658, 0.8676, 0.9250, 0.9584, 0.9742]) * 100

x_labels = ['nprobe=1', 'nprobe=2', 'nprobe=4',\
    'nprobe=8', 'nprobe=16', 'nprobe=32', 'nprobe=64', 'nprobe=128']

x = np.arange(len(x_labels))  # the label locations
width = 0.3  # the width of the bars    

# fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 2))

fig, ax0 = plt.subplots(1, 1, figsize=(8, 2))
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

ax0.legend([rects, line], ["QPS", "Recall"], facecolor='white', framealpha=1, frameon=True, loc=(0.83, 0.55), fontsize=legend_font, ncol=1)


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
ax0.set(ylim=[0, np.amax(y_QPS) * 1.1])

plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./out_img/cpu_throughput_recall_experiment_4_nprobe.png', transparent=False, dpi=200, bbox_inches="tight")
plt.show()
