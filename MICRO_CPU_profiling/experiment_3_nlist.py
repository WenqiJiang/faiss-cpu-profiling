"""
Evaluating the effect of nlist given the fixed nprobe

Note: if using perf to profile the program, use sudo to run the commands (already
    hardcoded in this script), make sure the user have sudo access

Example Usage:
        python experiment_3_nlist.py --dbname SIFT1000M --topK 100 --nprobe 16 --qbs 10000 --repeat_time 1 \
          --cpp_bin_dir /data/faiss-cpu-profiling/build/demos/bigann_search \
          --index_parent_dir /data/Faiss_experiments/trained_CPU_indexes/ \
          --gt_parent_dir /data/Faiss_experiments/bigann/gnd/ \
          --nprobe_dict_dir '../recall_info/cpu_recall_index_nprobe_pairs_SIFT1000M.pkl' --perf_enable 1
"""

from __future__ import print_function
import os
import sys
import time
import re
import pickle
import getpass
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type=str, default='SIFT1000M', help="dataset name, e.g., SIFT100M")
parser.add_argument('--topK', type=int, default=10, help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc")
parser.add_argument('--nprobe', type=int, default=16, help="number of cells to search in")
parser.add_argument('--qbs', type=int, default=10000, help="batch size")
parser.add_argument('--repeat_time', type=int, default=1, help="repeat_time of the 10000 queries, higher repeat time typically has a better stability")
parser.add_argument('--cpp_bin_dir', type=str, default='/data/faiss-cpu-profiling/build/demos/bigann_search', help="c++ search binary")
parser.add_argument('--index_parent_dir', type=str, default='/data/Faiss_experiments/trained_CPU_indexes/', help="parent directory of index storage")
parser.add_argument('--gt_parent_dir', type=str, default='/data/Faiss_experiments/bigann/gnd/', help="parent directory of ground truth")
parser.add_argument('--nprobe_dict_dir', type=str, default='../recall_info/cpu_recall_index_nprobe_pairs_SIFT1000M.pkl', help="recall dictionary, stores the min nprobe to achieve certain recall")

parser.add_argument('--perf_enable', type=int, default=1, help="whether to profile by perf")

args = parser.parse_args()
dbname = args.dbname
topK = args.topK
nprobe = args.nprobe
qbs = args.qbs
repeat_time = args.repeat_time
cpp_bin_dir = args.cpp_bin_dir
index_parent_dir = args.index_parent_dir
gt_parent_dir = args.gt_parent_dir
nprobe_dict_dir = args.nprobe_dict_dir
perf_enable = args.perf_enable

assert qbs == 10000, "Currently the c++ search program only support batch size = 10000"


out_dir = "result_experiment_3_nlist"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

logname = "./{out_dir}/out_{dbname}_K_{topK}_nprobe_{nprobe}_qbs_{qbs}".format(
    out_dir=out_dir, dbname=dbname, topK=topK, nprobe=nprobe, qbs=qbs)
if os.path.exists(logname):
    os.remove(logname)

gt_dir = None
if dbname == 'SIFT1M':
    gt_dir = os.path.join(gt_parent_dir, 'idx_1M.ivecs')
elif dbname == 'SIFT10M':
    gt_dir = os.path.join(gt_parent_dir, 'idx_10M.ivecs')
elif dbname == 'SIFT100M':
    gt_dir = os.path.join(gt_parent_dir, 'idx_100M.ivecs')
elif dbname == 'SIFT1000M':
    gt_dir = os.path.join(gt_parent_dir, 'idx_1000M.ivecs')
else:
    print("ERROR: unknown dataset")
    raise ValueError

index_keys = ['IVF1024,PQ16', 'IVF2048,PQ16', 'IVF4096,PQ16', 'IVF8192,PQ16', 'IVF16384,PQ16', 'IVF32768,PQ16', 'IVF65536,PQ16', \
    'OPQ16,IVF1024,PQ16', 'OPQ16,IVF2048,PQ16', 'OPQ16,IVF4096,PQ16', 'OPQ16,IVF8192,PQ16', 'OPQ16,IVF16384,PQ16', 'OPQ16,IVF32768,PQ16', 'OPQ16,IVF65536,PQ16']

for index_key in index_keys:

    os.system('echo ==== {index_key} ==== >> {logname}'.format(index_key=index_key, logname=logname))

    index_sub_dir = 'bench_cpu_{dbname}_{index_key}/{dbname}_{index_key}_populated.index'.format(dbname=dbname, index_key=index_key)
    index_dir = os.path.join(index_parent_dir, index_sub_dir)
    # Usage: ./binary index_dir gt_dir topK nprobe
    cmd = "{cpp_bin_dir} {index_dir} {gt_dir} {topK} {nprobe} {repeat_time} >> {logname}".format(
        cpp_bin_dir=cpp_bin_dir, index_dir=index_dir, gt_dir=gt_dir, topK=topK, nprobe=nprobe, repeat_time=repeat_time, logname=logname)

    if not perf_enable:
        print(cmd)
        os.system(cmd)
    else:
        cmd_prefix = "perf record -v -g -F 99 "
        cmd_prof = "sudo " + cmd_prefix + cmd
        print(cmd_prof)
        os.system(cmd_prof)

        # generate the perf.out, i.e., the trace of each sample

        reportname = "./{out_dir}/perf.out_{dbname}_{index_key}_K_{topK}_nprobe_{nprobe}_qbs_{qbs}".format(
            out_dir=out_dir, dbname=dbname, index_key=index_key, topK=topK, nprobe=nprobe,qbs=qbs)

        cmd_stats = "sudo perf script > {reportname}".format(reportname=reportname)
        os.system(cmd_stats)
        username = getpass.getuser()
        os.system("sudo chown {username} {reportname}".format(username=username, reportname=reportname))
        os.system("sudo rm perf.data perf.data.old")
