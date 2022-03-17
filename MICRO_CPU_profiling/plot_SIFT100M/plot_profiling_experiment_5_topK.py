import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os

from analyze_perf import group_perf_by_events, filter_events_after_timestamp, \
    classify_events_by_stages, get_percentage
from profiling_stages import draw_profiling_plot


x_labels = ['OPQ16,IVF65536\ntopK=1', \
    'OPQ16,IVF65536\ntopK=10', \
    'OPQ16,IVF65536\ntopK=20', \
    'OPQ16,IVF65536\ntopK=50', \
    'OPQ16,IVF65536\ntopK=100', \
    'OPQ16,IVF65536\ntopK=200', \
    'OPQ16,IVF65536\ntopK=500', \
    'OPQ16,IVF65536\ntopK=1000']

file_prefixes = [ \
    'perf.out_SIFT100M_OPQ16,IVF65536,PQ16_K_1_nprobe_64_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF65536,PQ16_K_10_nprobe_64_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF65536,PQ16_K_20_nprobe_64_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF65536,PQ16_K_50_nprobe_64_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF65536,PQ16_K_100_nprobe_64_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF65536,PQ16_K_200_nprobe_64_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF65536,PQ16_K_500_nprobe_64_qbs_10000', \
    'perf.out_SIFT100M_OPQ16,IVF65536,PQ16_K_1000_nprobe_64_qbs_10000']

assert len(x_labels) == len(file_prefixes)

path_prefixes = []
for p in file_prefixes:
    path_prefixes.append(os.path.join('../result_experiment_5_topK', p))


# time range of the search function, according to the search log, e.g.,
    # time_bias_start = 135.656
    # time_bias_end = 200.659
time_ranges = [ # pair of (time_bias_start, time_bias_end)
    # ==== topK=1 ====
    (5.965, 7.150),
    # ==== topK=10 ====
    (5.945, 7.091),
    # ==== topK=20 ====
    (5.871, 6.998),
    # ==== topK=50 ====
    (5.858, 7.071),
    # ==== topK=100 ====
    (6.774, 8.057),
    # ==== topK=200 ====
    (5.996, 7.285),
    # ==== topK=500 ====
    (5.878, 7.347),
    # ==== topK=1000 ====
    (5.853, 7.724)]

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
    t_1_4, t_5, t_6, t_other = classify_events_by_stages(filtered_events, track_non_faiss_func=False, remove_unrecognized_faiss_function=False)
    p_1_4, p_5, p_6, p_other = get_percentage(t_1_4, t_5, t_6, t_other)
    profile_perc_array.append([p_1_4, p_5, p_6, p_other])

y_stage_1_4 = [r[0] for r in profile_perc_array]
y_stage_5 = [r[1] for r in profile_perc_array]
y_stage_6 = [r[2] for r in profile_perc_array]
y_other = [r[3] for r in profile_perc_array]

draw_profiling_plot(x_labels, y_stage_1_4, y_stage_5, y_stage_6, y_other, 'cpu_profile_experiment_5_topK_SIFT100M', x_tick_rotation=45)
