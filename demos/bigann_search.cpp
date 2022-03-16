/*
  Usage: ./binary index_dir gt_dir topK nprobe (optional repeat_time)
  Example Usage:
    ./build/demos/bigann_search /data/trained_CPU_indexes_python/bench_cpu_SIFT100M_OPQ16,IVF65536,PQ16/SIFT100M_OPQ16,IVF65536,PQ16_populated.index /data/Faiss_experiments/bigann/gnd/idx_100M.ivecs 100 64 1
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

// Wenqi
#include <faiss/IndexIVFPQ.h>
#include <chrono>

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

// WENQI: load bvecs (1 int + uint8 * 128 per vec) in memory, as float format
float* bvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
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
    assert(sz % (d + 4) == 0 || !"weird file size");
    size_t n = sz / (d + 4);

    *d_out = d;
    *n_out = n;
    unsigned char* x = new unsigned char[n * (d + 4)];
    size_t nr = fread(x, sizeof(unsigned char), n * (d + 4), f);
    assert(nr == n * (d + 4) || !"could not read whole file");

    float* vec = new float[n * d];
    // remove header and convert uint8 to float
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            vec[i * d + j] = (float) x[i * (d + 4) + 4 + j]; 
        }
    }
    delete []x;

    fclose(f);
    return vec;
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
    using milli = std::chrono::milliseconds;
    auto t0 = std::chrono::high_resolution_clock::now();
    //double t0 = elapsed();

    if (argc < 5) {
        printf("Usage: ./binary index_dir gt_dir topK nprobe (optional repeat_time) \nExit\n");
        exit(1);
    }

    std::string index_dir = argv[1];
    // std::string index_dir = "/home/ubuntu/trained_CPU_indexes_C/SIFT1M_IVF1024,PQ16_populated_index";
    std::string gt_dir = argv[2];
    // std::string gt_dir = "/data/Faiss_experiments/bigann/gnd/idx_1M.ivecs";
    int k = std::stoi(argv[3]);
    int nprobe = std::stoi(argv[4]);
    int repeat_time = 1; // repeat the 10000 queries for N times to increase the performance measurement precision
    if (argc >= 6) { repeat_time = std::stoi(argv[5]); }

    // faiss::IndexIVFPQ* index = (faiss::IndexIVFPQ*) faiss::read_index(index_dir.c_str());
    faiss::Index* index = faiss::read_index(index_dir.c_str());
    // faiss::Index* index = faiss::read_index("/home/ubuntu/trained_CPU_indexes_python/bench_cpu_SIFT1M_IMI2x8,PQ16/SIFT1M_IMI2x8,PQ16_populated.index");

    // printf("imbalance factor of the index: %f\n", index -> invlists -> imbalance_factor());

    const size_t bigger_than_cachesize = 10 * 1024 * 1024;
    long *p = new long[bigger_than_cachesize];
    // Flush cacheline
    for(int i = 0; i < bigger_than_cachesize; i++)
    {
       p[i] = rand();
    }

    size_t d = 128;
    size_t nq;
    float* xq;

    {
        printf("[%.3f s] Loading queries\n", std::chrono::duration_cast<milli>(std::chrono::high_resolution_clock::now() - t0).count() / 1000.0);
        //printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = bvecs_read("/data/Faiss_experiments/bigann/bigann_query.bvecs", &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }
/*
    printf("content of the first query:\n");
    for (int i = 0; i < d; i++) {
        printf("%f\t", xq[i]);
    }
*/
    size_t k_max;                // topK of results per query in the GT
    faiss::Index::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               std::chrono::duration_cast<milli>(std::chrono::high_resolution_clock::now() - t0).count() / 1000.0,
               //elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int* gt_int = ivecs_read(gt_dir.c_str(), &k_max, &nq2);
printf("k=%d k_max=%d\n", k, k_max);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::Index::idx_t[k_max * nq];
        for (int i = 0; i < k_max * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }

    // Result of the auto-tuning
    std::string selected_params = std::string("nprobe=") + std::to_string(nprobe);

    { // Use the found configuration to perform a search

        faiss::ParameterSpace params;

        printf("[%.3f s] Setting parameter configuration \"%s\" on index\n",
               std::chrono::duration_cast<milli>(std::chrono::high_resolution_clock::now() - t0).count() / 1000.0,
               //elapsed() - t0,
               selected_params.c_str());

        params.set_index_parameters(index, selected_params.c_str());

        printf("[%.3f s] Perform a search on %ld queries\n",
               std::chrono::duration_cast<milli>(std::chrono::high_resolution_clock::now() - t0).count() / 1000.0,
               //elapsed() - t0,
               nq);

        // output buffers
        faiss::Index::idx_t* I = new faiss::Index::idx_t[nq * k];
        float* D = new float[nq * k];
	
	// WENQI: use more iterations to help perf record performance
        auto t_before_search = std::chrono::high_resolution_clock::now();
        //double t_before_search = elapsed();
        for (int rt = 0; rt < repeat_time; rt++) {
            // Flush cacheline
            for(int i = 0; i < bigger_than_cachesize; i++)
            {  
       		p[i] = 0;
   	    }
            index->search(nq, xq, k, D, I); 
        }

        double t_search = std::chrono::duration_cast<milli>(std::chrono::high_resolution_clock::now() - t_before_search).count() / 1000.0;
        //double t_search = elapsed() - t_before_search;
        double QPS = repeat_time * ((double) nq) / t_search;
        printf("Search complete, takes [%.3f s],  QPS=%.3f\n", t_search, QPS);
        printf("[%.3f s] Compute recalls\n", std::chrono::duration_cast<milli>(std::chrono::high_resolution_clock::now() - t0).count() / 1000.0);
        //printf("[%.3f s] Compute recalls\n", elapsed() - t0);

        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0, n_k = 0;
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
                    n_k++;
                }
		else {
//		    printf("gt: %d\treturned: %d\n", gt_nn, I[i * k + j]);
		}
            }
        }
        if (k >= 1) printf("R@1 = %.4f\n", n_1 / float(nq));
        if (k >= 10) printf("R@10 = %.4f\n", n_10 / float(nq));
        if (k >= 100) printf("R@100 = %.4f\n", n_100 / float(nq));
        if (k != 1 && k != 10 && k != 100) printf("R@%d = %.4f\n", k, n_k / float(nq));

        delete[] I;
        delete[] D;
    }

    delete[] xq;
    delete[] gt;
    delete index;
    return 0;
}
