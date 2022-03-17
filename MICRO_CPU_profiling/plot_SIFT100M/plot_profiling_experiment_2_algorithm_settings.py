import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os

from analyze_perf import group_perf_by_events, filter_events_after_timestamp, \
    classify_events_by_stages, get_percentage
from profiling_stages import draw_profiling_plot


x_labels = ['IVF1024\nnprobe=12', 'OPQ,IVF1024\nnprobe=11', \
    'IVF2048\nnprobe=15', 'OPQ,IVF2048\nnprobe=14', \
    'IVF4096\nnprobe=20', 'OPQ,IVF4096\nnprobe=19', \
    'IVF8192\nnprobe=26', 'OPQ,IVF8192\nnprobe=25', \
    'IVF16384\nnprobe=35', 'OPQ,IVF16384\nnprobe=33', \
    'IVF32768\nnprobe=49', 'OPQ,IVF32768\nnprobe=47', \
    'IVF65536\nnprobe=65', 'OPQ,IVF65536\nnprobe=64', \
    'IVF131072\nnprobe=84', 'OPQ,IVF131072\nnprobe=79', \
    'IVF262144\nnprobe=115', 'OPQ,IVF262144\nnprobe=116']

file_prefixes = [ \
    'perf.out_SIFT100M_IVF1024,PQ16_R@100=0.95_nprobe_12_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF1024,PQ16_R@100=0.95_nprobe_11_qbs_10000', \
    'perf.out_SIFT100M_IVF2048,PQ16_R@100=0.95_nprobe_15_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF2048,PQ16_R@100=0.95_nprobe_14_qbs_10000', \
    'perf.out_SIFT100M_IVF4096,PQ16_R@100=0.95_nprobe_20_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF4096,PQ16_R@100=0.95_nprobe_19_qbs_10000', \
    'perf.out_SIFT100M_IVF8192,PQ16_R@100=0.95_nprobe_26_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF8192,PQ16_R@100=0.95_nprobe_25_qbs_10000', \
    'perf.out_SIFT100M_IVF16384,PQ16_R@100=0.95_nprobe_35_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF16384,PQ16_R@100=0.95_nprobe_33_qbs_10000', \
    'perf.out_SIFT100M_IVF32768,PQ16_R@100=0.95_nprobe_49_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF32768,PQ16_R@100=0.95_nprobe_47_qbs_10000', \
    'perf.out_SIFT100M_IVF65536,PQ16_R@100=0.95_nprobe_65_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF65536,PQ16_R@100=0.95_nprobe_64_qbs_10000', \
    'perf.out_SIFT100M_IVF131072,PQ16_R@100=0.95_nprobe_84_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF131072,PQ16_R@100=0.95_nprobe_79_qbs_10000', \
    'perf.out_SIFT100M_IVF262144,PQ16_R@100=0.95_nprobe_115_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF262144,PQ16_R@100=0.95_nprobe_116_qbs_10000']

assert len(x_labels) == len(file_prefixes)

path_prefixes = []
for p in file_prefixes:
    path_prefixes.append(os.path.join('../result_experiment_2_algorithm_settings', p))

# time range of the search function, according to the search log, e.g.,
    # time_bias_start = 135.656
    # time_bias_end = 200.659
time_ranges = [ # pair of (time_bias_start, time_bias_end)
    # ==== IVF1024,PQ16 ====
    (2.788, 9.451),
    # ==== OPQ16,IVF1024,PQ16 ====
    (2.788, 9.412),
    # ==== IVF2048,PQ16 ====
    (2.905, 7.661),
    # ==== OPQ16,IVF2048,PQ16 ====
    (2.912, 7.000),
    # ==== IVF4096,PQ16 ====
    (2.947, 6.895),
    # ==== OPQ16,IVF4096,PQ16 ====
    (3.054, 17.223),
    # ==== IVF8192,PQ16 ====
    (14.464, 17.223),
    # ==== OPQ16,IVF8192,PQ16 ====
    (14.482, 17.258),
    # ==== IVF16384,PQ16 ====
    (3.852, 5.546),
    # ==== OPQ16,IVF16384,PQ16 ====
    (3.909, 5.679),
    # ==== IVF32768,PQ16 ====
    (4.857, 6.198),
    # ==== OPQ16,IVF32768,PQ16 ====
    (4.844, 6.283),
    # ==== IVF65536,PQ16 ====
    (6.794, 8.216),
    # ==== OPQ16,IVF65536,PQ16 ====
    (6.774, 8.057),
    # ==== IVF131072,PQ16 ====
    (10.468, 11.859),
    # ==== OPQ16,IVF131072,PQ16 ====
    (10.460, 11.814),
    # ==== IVF262144,PQ16 ====
    (16.478, 34.546),
    # ==== OPQ16,IVF262144,PQ16 ====
    (16.121, 35.389) ]

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
    t_1_4, t_5, t_6, t_other = classify_events_by_stages(filtered_events, track_non_faiss_func=False)
    p_1_4, p_5, p_6, p_other = get_percentage(t_1_4, t_5, t_6, t_other)
    profile_perc_array.append([p_1_4, p_5, p_6, p_other])

y_stage_1_4 = [r[0] for r in profile_perc_array]
y_stage_5 = [r[1] for r in profile_perc_array]
y_stage_6 = [r[2] for r in profile_perc_array]
y_other = [r[3] for r in profile_perc_array]

draw_profiling_plot(x_labels, y_stage_1_4, y_stage_5, y_stage_6, y_other, 'cpu_profile_experiment_2_algorithm_settings_SIFT100M', x_tick_rotation=70)

