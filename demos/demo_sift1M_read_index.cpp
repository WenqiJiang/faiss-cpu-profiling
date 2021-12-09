/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
    double t0 = elapsed();

    std::string index_dir = "/home/ubuntu/trained_CPU_indexes_C/SIFT1M_IVF1024,PQ16_populated_index";
    index_dir = argv[1];

    faiss::Index* index = faiss::read_index(index_dir.c_str());
    // faiss::Index* index = faiss::read_index("/home/ubuntu/trained_CPU_indexes_python/bench_cpu_SIFT1M_IMI2x8,PQ16/SIFT1M_IMI2x8,PQ16_populated.index");
    // faiss::Index* index = faiss::read_index("/home/ubuntu/index/SIFT1M_IVF1024,PQ16_populated_index");
    //faiss::Index* index = faiss::read_index("/home/ubuntu/trained_CPU_indexes/bench_cpu_SIFT1M_IVF1024,PQ16/SIFT1M_IVF1024,PQ16_populated.index");;

    size_t d = 128;
    size_t nq;
    float* xq;

    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }

    size_t k = 100;                // topK of results per query
    size_t k_max;                // topK of results per query in the GT
    faiss::Index::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int* gt_int = ivecs_read("sift1M/sift_groundtruth.ivecs", &k_max, &nq2);
//k=k_max;
printf("k=%d k_max=%d\n", k, k_max);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::Index::idx_t[k_max * nq];
        for (int i = 0; i < k_max * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }

    // Result of the auto-tuning
    std::string selected_params = "nprobe=64";

    { // Use the found configuration to perform a search

        faiss::ParameterSpace params;

        printf("[%.3f s] Setting parameter configuration \"%s\" on index\n",
               elapsed() - t0,
               selected_params.c_str());

        params.set_index_parameters(index, selected_params.c_str());

        printf("[%.3f s] Perform a search on %ld queries\n",
               elapsed() - t0,
               nq);

        // output buffers
        faiss::Index::idx_t* I = new faiss::Index::idx_t[nq * k];
        float* D = new float[nq * k];
	
	// WENQI: use more iterations to help perf record performance
	for (int search_cnt = 0; search_cnt < 100; search_cnt++) {
        index->search(nq, xq, k, D, I);
        }

        double t_search = elapsed() - t0;
        double QPS = ((double) nq) / t_search;
        printf("[%.3f s] Search complete, QPS=%.3f\n", t_search, QPS);
        printf("[%.3f s] Compute recalls\n", elapsed() - t0);

        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for (int i = 0; i < nq; i++) {
            int gt_nn = gt[i * k_max];
            for (int j = 0; j < k; j++) {
                if (I[i * k + j] == gt_nn) {
                    if (j < 1)
                        n_1++;
                    if (j < 10)
                        n_10++;
                    if (j < 100)
                        n_100++;
                }
		else {
//		    printf("gt: %d\treturned: %d\n", gt_nn, I[i * k + j]);
		}
            }
        }
        printf("R@1 = %.4f\n", n_1 / float(nq));
        printf("R@10 = %.4f\n", n_10 / float(nq));
        printf("R@100 = %.4f\n", n_100 / float(nq));

        delete[] I;
        delete[] D;
    }

    delete[] xq;
    delete[] gt;
    delete index;
    return 0;
}
