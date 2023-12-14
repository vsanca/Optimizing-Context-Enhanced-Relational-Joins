//
// Created by sanca on 9/22/23.
//

#ifndef J2V_TESTS_H
#define J2V_TESTS_H

#include "scan.h"
#include "join2vec.h"

class Tests{
public:
    using res_pair = std::tuple<size_t, size_t>;

    Tests();
    void scan_vector(size_t byte_width, size_t vector_size, size_t count);
    void scan_columns(size_t byte_width, size_t count, int iter=0);
    void scan_columns2(size_t count, int repeat, vector<int> offsets);
    void scan_columns_rand(size_t count, int repeat, vector<int> offsets);
    void scan_columns_rand2(size_t count, int repeat, vector<int> offsets);
    void scan_columns_rand3(size_t count, int repeat, vector<int> offsets);
    void scan_vector_columns();

    void cache_cosine_sim(size_t arrayA, size_t vectorD, float thr, bool SIMD);
    void cache_cosine_sim_vec(size_t arrayA, size_t vectorD, float thr, bool SIMD);
    void cache_cosine_sim1(size_t arrayA, size_t vectorD, float thr, bool SIMD);
    void cache_cosine_sim11(size_t arrayA, float thr, bool SIMD);
    void cache_cosine_sim_block1(size_t arrayA, bool SIMD);

    void cache_cosine_sim_blas(size_t arrayA, size_t vectorD, size_t batch, float thr);
    void blas_mult_only(size_t arrayA, size_t arrayB, size_t vectorD, size_t batch); // TODO: implement
    void cache_cosine_sim_blas_normalize_only(size_t arrayA, size_t vectorD, size_t batch, float thr);

    void cache_cosine_sim_blas1(size_t arrayA, size_t vectorD, size_t batch, float thr);
    void cache_cosine_sim_blas_normalize_only1(size_t arrayA, size_t vectorD, size_t batch, float thr);

    /**
     * 64-bit version, TODO: cleanup - just template it.
     */
    void cache_cosine_sim64(size_t arrayA, size_t vectorD, float thr, bool SIMD);
    void cache_cosine_sim_blas64(size_t arrayA, size_t vectorD, size_t batch, float thr);
    void cache_cosine_sim_blas_normalize_only64(size_t arrayA, size_t vectorD, size_t batch, float thr);

    void join_BLAS(size_t arrayA, size_t arrayB, size_t vectorD, float thr);
    void join_BLAS_batch(size_t arrayA, size_t arrayB, size_t vectorD, float thr, pair<size_t, size_t> batch);

    map<string, vector<long>> resCnt;
};

#endif //J2V_TESTS_H
