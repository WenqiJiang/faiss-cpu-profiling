import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os

from analyze_perf import group_perf_by_events, filter_events_after_timestamp, \
    classify_events_by_stages, get_percentage
from profiling_stages import draw_profiling_plot


x_labels = ['IVF65536\ntopK=1', \
    'IVF65536\ntopK=10', \
    'IVF65536\ntopK=20', \
    'IVF65536\ntopK=50', \
    'IVF65536\ntopK=100', \
    'IVF65536\ntopK=200', \
    'IVF65536\ntopK=500', \
    'IVF65536\ntopK=1000']

file_prefixes = [ \
    'perf.out_SIFT1000M_IVF65536,PQ16_K_1_nprobe_51_qbs_10000', \
    'perf.out_SIFT1000M_IVF65536,PQ16_K_10_nprobe_51_qbs_10000', \
    'perf.out_SIFT1000M_IVF65536,PQ16_K_20_nprobe_51_qbs_10000', \
    'perf.out_SIFT1000M_IVF65536,PQ16_K_50_nprobe_51_qbs_10000', \
    'perf.out_SIFT1000M_IVF65536,PQ16_K_100_nprobe_51_qbs_10000', \
    'perf.out_SIFT1000M_IVF65536,PQ16_K_200_nprobe_51_qbs_10000', \
    'perf.out_SIFT1000M_IVF65536,PQ16_K_500_nprobe_51_qbs_10000', \
    'perf.out_SIFT1000M_IVF65536,PQ16_K_1000_nprobe_51_qbs_10000']

assert len(x_labels) == len(file_prefixes)

path_prefixes = []
for p in file_prefixes:
    path_prefixes.append(os.path.join('result_experiment_5_topK', p))


# time range of the search function, according to the search log, e.g.,
    # time_bias_start = 135.656
    # time_bias_end = 200.659
time_ranges = [ # pair of (time_bias_start, time_bias_end)
    # ==== topK=1 ====
    (29.905, 70.366),
    # ==== topK=10 ====
    (29.049, 68.554),
    # ==== topK=20 ====
    (28.837, 68.518),
    # ==== topK=50 ====
    (29.748, 70.364),
    # ==== topK=100 ====
    (29.285, 68.431),
    # ==== topK=200 ====
    (28.966, 69.081),
    # ==== topK=500 ====
    (29.086, 68.934),
    # ==== topK=1000 ====
    (29.727, 71.050)]

# Stage 1: OPQ
# Stage 2: vector quantizer
# Stage 3: select centroids
# Stage 4: construct distance LUT
# Stage 5: PQ code scan
# Stage 6: collect topK results

profile_perc_array = []
# example_profile_array = [
#     # 100M, 1
#     [8.606278140845747, 0.11607633274229297, 3.3378707089447355, 78.57136070072978, 9.368414116737446], \
#     # 100M, 10
#     [32.7008185883583, 0.5164703077320218, 4.674772663594282, 33.70847203114799, 28.399466409167403]
#     ]

for i in range(len(path_prefixes)): 
    print("Processing {}".format(path_prefixes[i]))
    all_events = group_perf_by_events(path_prefixes[i])
    time_bias_start, time_bias_end = time_ranges[i][0], time_ranges[i][1]
    filtered_events = filter_events_after_timestamp(all_events, time_bias_start, time_bias_end)
    t_1_2, t_3, t_4, t_5, t_6, t_other = classify_events_by_stages(filtered_events, track_non_faiss_func=False)
    p_1_2, p_3, p_4, p_5, p_6, p_other = get_percentage(t_1_2, t_3, t_4, t_5, t_6, t_other)
    profile_perc_array.append([p_1_2, p_3, p_4, p_5, p_6, p_other])

y_stage_1_2 = [r[0] for r in profile_perc_array]
y_stage_3 = [r[1] for r in profile_perc_array]
y_stage_4 = [r[2] for r in profile_perc_array]
y_stage_5 = [r[3] for r in profile_perc_array]
y_stage_6 = [r[4] for r in profile_perc_array]
y_other = [r[5] for r in profile_perc_array]

draw_profiling_plot(x_labels, y_stage_1_2, y_stage_3, y_stage_4, y_stage_5, y_stage_6, y_other, 'cpu_profile_experiment_5_topK', x_tick_rotation=45)
