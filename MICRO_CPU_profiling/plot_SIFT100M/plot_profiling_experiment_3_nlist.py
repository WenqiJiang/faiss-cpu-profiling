import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os

from analyze_perf import group_perf_by_events, filter_events_after_timestamp, \
    classify_events_by_stages, get_percentage
from profiling_stages import draw_profiling_plot


x_labels = ['IVF1024\nnprobe=16', \
    'IVF2048\nnprobe=16', \
    'IVF4096\nnprobe=16', \
    'IVF8192\nnprobe=16', \
    'IVF16384\nnprobe=16', \
    'IVF32768\nnprobe=16', \
    'IVF65536\nnprobe=16', \
    'IVF131072\nnprobe=16', \
    'IVF262144\nnprobe=16']

file_prefixes = [ \
    'perf.out_SIFT100M_IVF1024,PQ16_K_100_nprobe_16_qbs_10000', \
    'perf.out_SIFT100M_IVF2048,PQ16_K_100_nprobe_16_qbs_10000', \
    'perf.out_SIFT100M_IVF4096,PQ16_K_100_nprobe_16_qbs_10000', \
    'perf.out_SIFT100M_IVF8192,PQ16_K_100_nprobe_16_qbs_10000', \
    'perf.out_SIFT100M_IVF16384,PQ16_K_100_nprobe_16_qbs_10000', \
    'perf.out_SIFT100M_IVF32768,PQ16_K_100_nprobe_16_qbs_10000', \
    'perf.out_SIFT100M_IVF65536,PQ16_K_100_nprobe_16_qbs_10000', \
    'perf.out_SIFT100M_IVF131072,PQ16_K_100_nprobe_16_qbs_10000', \
    'perf.out_SIFT100M_IVF262144,PQ16_K_100_nprobe_16_qbs_10000']

assert len(x_labels) == len(file_prefixes)

path_prefixes = []
for p in file_prefixes:
    path_prefixes.append(os.path.join('../result_experiment_3_nlist', p))

# time range of the search function, according to the search log, e.g.,
    # time_bias_start = 135.656
    # time_bias_end = 200.659
time_ranges = [ # pair of (time_bias_start, time_bias_end)
    # ==== IVF1024,PQ16 ====
    (2.807 , 11.893),
    # ==== IVF2048,PQ16 ====
    (2.915, 8.534),
    # ==== IVF4096,PQ16 ====
    (3.018, 5.739),
    # ==== IVF8192,PQ16 ====
    (3.364, 4.672),
    # ==== IVF16384,PQ16 ====
    (3.937, 4.827),
    # ==== IVF32768,PQ16 ====
    (4.824, 5.783),
    # ==== IVF65536,PQ16 ====
    (6.821, 7.675),
    # ==== IVF131072,PQ16 ====
    (10.458, 11.656),
    # ==== IVF262144,PQ16 ====
    (3.985, 20.681),
    # ==== OPQ16,IVF1024,PQ16 ====
    (2.722, 11.790),
    # ==== OPQ16,IVF2048,PQ16 ====
    (2.933, 7.308),
    # ==== OPQ16,IVF4096,PQ16 ====
    (3.033, 5.811),
    # ==== OPQ16,IVF8192,PQ16 ====
    (3.309, 4.886),
    # ==== OPQ16,IVF16384,PQ16 ====
    (3.872, 4.932),
    # ==== OPQ16,IVF32768,PQ16 ====
    (4.847, 5.612),
    # ==== OPQ16,IVF65536,PQ16 ====
    (6.733, 7.484),
    # ==== OPQ16,IVF131072,PQ16 ====
    (10.298, 11.359),
    # ==== OPQ16,IVF262144,PQ16 ====
    (4.017, 21.640)]

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

draw_profiling_plot(x_labels, y_stage_1_4, y_stage_5, y_stage_6, y_other, 'cpu_profile_experiment_3_nlist_SIFT100M', x_tick_rotation=45)

