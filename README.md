# Faiss-CPU-Profiling

This repo is forked from Faiss. We use Perf to profile the performance of a range of index settings.

## Build from Source

```
git clone https://github.com/WenqiJiang/faiss-cpu-profiling
cd faiss-cpu-profiling

# install Anaconda, and create a env using python3.7
conda create -n faiss_build python=3.7
conda activate faiss_build

install cmake: https://vitux.com/how-to-install-cmake-on-ubuntu/
install openblas:
sudo apt-get install libopenblas-dev
if install python interface, install swig: conda install -c anaconda swig

install faiss: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
build command (some of these options can be removed):
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 
# for GPU, the options can be (V100 has compute capability 7.0): 
cmake -B build . -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 -DCUDAToolkit_ROOT=/usr/local/cuda-11.4/ -DCMAKE_CUDA_ARCHITECTURES="70"

# follow the rest install instruction
# demo test is also introduced in the install part
make -C build -j faiss
make -C build -j swigfaiss
(cd build/faiss/python && python setup.py install)
make -C build install
make -C build test

after adding a new file to demo:
(1) add its to demo/CMakefile list
(2) then rerun build
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 -DFAISS_ENABLE_PYTHON=OFF 
then do the build:
make -C build demo_sift1M_read_index

include #include <faiss/index_io.h> when using read_index and write_index

# building demo c++ programs
cd /mnt/scratch/wenqi/faiss/
make -C build demo_sift1M_read_index
./build/demos/demo_sift1M_read_index ../trained_CPU_indexes_C/SIFT1M_IVF1024,PQ16_populated_index 
```

## CPU Profiling Scripts

```
cd faiss-cpu-profiling
make -C build bigann_search
cd MICRO_CPU_profiling/

# run all the scripts, e.g., for 100M
# experiment 4~5's index depends on which index performs the best in experiment 2
python experiment_2_algorithm_settings.py --dbname SIFT100M --topK 100 --recall_goal 0.95 --qbs 10000 --repeat_time 1 \
      --cpp_bin_dir /home/ubuntu/faiss-cpu-profiling/build/demos/bigann_search \
      --index_parent_dir /data/Faiss_experiments/trained_CPU_indexes/ \
      --gt_parent_dir /data/Faiss_experiments/bigann/gnd/ \
      --nprobe_dict_dir '../recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --perf_enable 1


python experiment_3_nlist.py --dbname SIFT100M --topK 100 --nprobe 16 --qbs 10000 --repeat_time 1 \
      --cpp_bin_dir /home/ubuntu/faiss-cpu-profiling/build/demos/bigann_search \
          --index_parent_dir /data/Faiss_experiments/trained_CPU_indexes/ \
          --gt_parent_dir /data/Faiss_experiments/bigann/gnd/ \
          --nprobe_dict_dir '../recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --perf_enable 1
     
python experiment_4_nprobe.py --dbname SIFT100M --index_key OPQ16,IVF65536,PQ16 --topK 100 --min_nprobe 1 --max_nprobe 128 --qbs 10000 --repeat_time 10 \
      --cpp_bin_dir /home/ubuntu/faiss-cpu-profiling/build/demos/bigann_search \
      --index_parent_dir /data/Faiss_experiments/trained_CPU_indexes/ \
      --gt_parent_dir /data/Faiss_experiments/bigann/gnd/ \
      --nprobe_dict_dir '../recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --perf_enable 1
      
python experiment_5_topK.py --dbname SIFT100M --index_key OPQ16,IVF65536,PQ16  --nprobe 51 --qbs 10000 --repeat_time 10 \
      --cpp_bin_dir /home/ubuntu/faiss-cpu-profiling/build/demos/bigann_search \
          --index_parent_dir /data/Faiss_experiments/trained_CPU_indexes/ \
          --gt_parent_dir /data/Faiss_experiments/bigann/gnd/ \
          --nprobe_dict_dir '../recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --perf_enable 1
          
# For SIFT 1000M
# experiment 4~5's index depends on which index performs the best in experiment 2
python experiment_2_algorithm_settings.py --dbname SIFT1000M --topK 100 --recall_goal 0.95 --qbs 10000 --repeat_time 1 \
      --cpp_bin_dir /home/ubuntu/faiss-cpu-profiling/build/demos/bigann_search \
      --index_parent_dir /data/Faiss_experiments/trained_CPU_indexes/ \
      --gt_parent_dir /data/Faiss_experiments/bigann/gnd/ \
      --nprobe_dict_dir '../recall_info/cpu_recall_index_nprobe_pairs_SIFT1000M.pkl' --perf_enable 1

python experiment_3_nlist.py --dbname SIFT1000M --topK 100 --nprobe 16 --qbs 10000 --repeat_time 1 \
      --cpp_bin_dir /home/ubuntu/faiss-cpu-profiling/build/demos/bigann_search \
          --index_parent_dir /data/Faiss_experiments/trained_CPU_indexes/ \
          --gt_parent_dir /data/Faiss_experiments/bigann/gnd/ \
          --nprobe_dict_dir '../recall_info/cpu_recall_index_nprobe_pairs_SIFT1000M.pkl' --perf_enable 1
     
python experiment_4_nprobe.py --dbname SIFT1000M --index_key OPQ16,IVF262144,PQ16 --topK 100 --min_nprobe 1 --max_nprobe 128 --qbs 10000 --repeat_time 10 \
      --cpp_bin_dir /home/ubuntu/faiss-cpu-profiling/build/demos/bigann_search \
      --index_parent_dir /data/Faiss_experiments/trained_CPU_indexes/ \
      --gt_parent_dir /data/Faiss_experiments/bigann/gnd/ \
      --nprobe_dict_dir '../recall_info/cpu_recall_index_nprobe_pairs_SIFT1000M.pkl' --perf_enable 1
      
python experiment_5_topK.py --dbname SIFT1000M --index_key OPQ16,IVF262144,PQ16  --nprobe 51 --qbs 10000 --repeat_time 10 \
      --cpp_bin_dir /home/ubuntu/faiss-cpu-profiling/build/demos/bigann_search \
          --index_parent_dir /data/Faiss_experiments/trained_CPU_indexes/ \
          --gt_parent_dir /data/Faiss_experiments/bigann/gnd/ \
          --nprobe_dict_dir '../recall_info/cpu_recall_index_nprobe_pairs_SIFT1000M.pkl' --perf_enable 1
    
# when plotting, note that: (1) the perf records the entire program, thus please remove the head & tail, the middle time can be viewed in the log generated in the result_xxxx folder

```
