import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os

from analyze_perf import group_perf_by_events, filter_events_after_timestamp, \
    classify_events_by_stages, get_percentage
from profiling_stages import draw_profiling_plot


x_labels = ['IVF1024', 'IVF1024\nw/ OPQ', \
    'IVF2048', 'IVF2048\nw/ OPQ', \
    'IVF4096', 'IVF4096\nw/ OPQ', \
    'IVF8192', 'IVF8192\nw/ OPQ', \
    'IVF16384', 'IVF16384\nw/ OPQ', \
    'IVF32768', 'IVF32768\nw/ OPQ', \
    'IVF65536', 'IVF65536\nw/ OPQ']

file_prefixes = [ \
    'perf.out_SIFT1000M_IVF1024,PQ16_R@100=0.95_nprobe_16_qbs_10000', \
    'perf.out_SIFT1000M_OPQ16,IVF1024,PQ16_R@100=0.95_nprobe_13_qbs_10000', \
    'perf.out_SIFT1000M_IVF2048,PQ16_R@100=0.95_nprobe_17_qbs_10000', \
    'perf.out_SIFT1000M_OPQ16,IVF2048,PQ16_R@100=0.95_nprobe_14_qbs_10000', \
    'perf.out_SIFT1000M_IVF4096,PQ16_R@100=0.95_nprobe_21_qbs_10000', \
    'perf.out_SIFT1000M_OPQ16,IVF4096,PQ16_R@100=0.95_nprobe_18_qbs_10000', \
    'perf.out_SIFT1000M_IVF8192,PQ16_R@100=0.95_nprobe_26_qbs_10000', \
    'perf.out_SIFT1000M_OPQ16,IVF8192,PQ16_R@100=0.95_nprobe_25_qbs_10000', \
    'perf.out_SIFT1000M_IVF16384,PQ16_R@100=0.95_nprobe_34_qbs_10000', \
    'perf.out_SIFT1000M_OPQ16,IVF16384,PQ16_R@100=0.95_nprobe_31_qbs_10000', \
    'perf.out_SIFT1000M_IVF32768,PQ16_R@100=0.95_nprobe_37_qbs_10000', \
    'perf.out_SIFT1000M_OPQ16,IVF32768,PQ16_R@100=0.95_nprobe_35_qbs_10000', \
    'perf.out_SIFT1000M_IVF65536,PQ16_R@100=0.95_nprobe_51_qbs_10000', \
    'perf.out_SIFT1000M_OPQ16,IVF65536,PQ16_R@100=0.95_nprobe_50_qbs_10000']

assert len(x_labels) == len(file_prefixes)

path_prefixes = []
for p in file_prefixes:
    path_prefixes.append(os.path.join('result_experiment_2_algorithm_settings', p))

# time range of the search function, according to the search log, e.g.,
    # time_bias_start = 135.656
    # time_bias_end = 200.659
time_ranges = [ # pair of (time_bias_start, time_bias_end)
    # ==== IVF1024,PQ16 ====
    (24.701, 109.128),
    # ==== OPQ16,IVF1024,PQ16 ====
    (135.656, 200.659),
    # ==== IVF2048,PQ16 ====
    (137.161, 181.207), 
    # ==== OPQ16,IVF2048,PQ16 ====
    (128.055, 164.959),
    # ==== IVF4096,PQ16 ====
    (24.460, 51.717),
    # ==== OPQ16,IVF4096,PQ16 ====
    (122.927, 146.346),
    # ==== IVF8192,PQ16 ====
    (24.548, 41.720),
    # ==== OPQ16,IVF8192,PQ16 ====
    (115.663, 132.339),
    # ==== IVF16384,PQ16 ====
    (119.551, 131.259), 
    # ==== OPQ16,IVF16384,PQ16 ====
    (132.458, 143.132),
    # ==== IVF32768,PQ16 ====
    (119.306, 126.174),
    # ==== OPQ16,IVF32768,PQ16 ====
    (130.724, 136.956),
    # ==== IVF65536,PQ16 ====
    (28.327, 32.994),
    # ==== OPQ16,IVF65536,PQ16 ====
    (135.239, 141.575)]

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

draw_profiling_plot(x_labels, y_stage_1_2, y_stage_3, y_stage_4, y_stage_5, y_stage_6, y_other, 'cpu_profile_experiment_2_algorithm_settings', x_tick_rotation=70)

