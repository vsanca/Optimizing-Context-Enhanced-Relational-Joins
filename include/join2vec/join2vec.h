//
// Created by sanca on 9/21/22.
//

#ifndef J2V_JOIN2VEC_H
#define J2V_JOIN2VEC_H

#include <string>
#include <vector>
#include "fasttext/fasttext.h"
#include <experimental/filesystem>
#include "Simd/SimdLib.h"
#include <mutex>
#include <bitset>

#include "util/TimeBlock.h"
#include "util/ThreadPool.h"
#include "join2vec/myVector.h"

#include "Eigen/Dense"
#include "Eigen/Core"

using namespace std;

typedef struct j2v_parameters_t
{
    int threads = 1;

    char *dataset_filename = NULL;
    char *model_filename = NULL;
    char *results_filename = NULL;
    char *timers_filename = NULL;

    double th = 0.8;

    bool verbose = false;
    bool preload_vectors = false;
    bool simd = false;
    bool cluster = false;
    bool cluster_prog = false;

    int dims = -1;
    int decimals = -1;

} j2v_parameters_t;

typedef struct j2v_result
{
    std::string w1;
    std::string w2;
    double score;
} j2v_result;

class Join2Vec{
private:
    string model_filename;

    vector<vector<float>> clusters;


    bool isModelLoaded = false;
    bool areWordVectorsLoaded = false;
    bool areWordsLoaded = false;

    size_t dimensions;

    void nested_loop_join(const vector<string>&, const vector<string>&, double threshold, float(Join2Vec::*f)(const float*, const float*,unsigned int));
    void nested_loop_join(const vector<myVector>&, const vector<myVector>&, double threshold, float(Join2Vec::*f)(const float*, const float*,unsigned int));    // data-parallel
    void nested_loop_join_(const vector<myVector>&, const vector<myVector>&, double threshold, float(Join2Vec::*f)(const float*, const float*,unsigned int));   // fetch-and-increment

    void nested_loop_cosine(const vector<myVector>&, const vector<myVector>&, double threshold, float(Join2Vec::*f)(const float*, const float*,unsigned int));    // data-parallel

    vector<vector<tuple<size_t, size_t>>> result;

    std::mutex inc;

public:

    fasttext::FastText model;

    Join2Vec(j2v_parameters_t param);
    Join2Vec(const std::string& model_path);

    void loadModel();
    std::vector<std::string> load_data(string filename);

    void join(vector<myVector>& R, std::vector<myVector>& S, double threshold, bool SIMD = true, bool equi_test=false);
    void join_cosine_only(vector<myVector>& R, std::vector<myVector>& S, double threshold, bool SIMD = true, bool equi_test=false);
    void join_(vector<myVector>& R, std::vector<myVector>& S, double threshold, bool SIMD = true, bool equi_test=false);
    void join(vector<string>& R, std::vector<string>& S, double threshold, bool SIMD = true);
    void join_matrix(vector<myVector>& R, std::vector<myVector>& S, double threshold, bool SIMD = true);
    void join_matrix(vector<vector<double>>& R, std::vector<vector<double>>& S, double threshold, bool SIMD = true);
    void join_matrix(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&Rm, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &Sm, double threshold, bool SIMD = true);
    void join_matrixP(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&Rm, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &Sm, double threshold, bool SIMD = true);


    vector<myVector> prefetch(vector<string>& R);

    vector<myVector> prefetchP(vector<string>& R);

    size_t getDimensions();

    using res_pair = std::tuple<size_t, size_t>;

    void cluster(size_t num_clusters);

    void clear_result() {
        result.clear();
    }

    typeof(result)& getResult(){
        return result;
    }

    vector<tuple<size_t, size_t>> getFlatResult(){
        vector<tuple<size_t, size_t>> ret;

        for(const auto el: result){
            for(const auto ell : el){
                ret.push_back(ell);
            }
        }

        return ret;
    }

    inline float cosine_similarity(float *A, float *B, unsigned int size)
    {
        /*float mul = 0.0;
        float d_a = 0.0;
        float d_b = 0.0;

        for (unsigned int i = 0; i < size; ++i)
        {
            mul += *A * *B;
            d_a += *A * *A;
            d_b += *B * *B;

            A++;
            B++;
        }

        if (d_a == 0.0f || d_b == 0.0f)
        {
            throw std::logic_error(
                    "cosine similarity is not defined whenever one or both "
                    "input vectors are zero-vectors.");
        }

        return mul / (sqrt(d_a) * sqrt(d_b));*/

        double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
        for(unsigned int i = 0; i < size; ++i) {
            dot += A[i] * B[i] ;
            denom_a += A[i] * A[i] ;
            denom_b += B[i] * B[i] ;
        }

        return dot / (sqrt(denom_a) * sqrt(denom_b));
    }

    inline float cosine_similarity(const float *A, const float *B, unsigned int size)
    {
        /*float mul = 0.0;
        float d_a = 0.0;
        float d_b = 0.0;

        for (unsigned int i = 0; i < size; ++i)
        {
            mul += *A * *B;
            d_a += *A * *A;
            d_b += *B * *B;

            A++;
            B++;
        }

        if (d_a == 0.0f || d_b == 0.0f)
        {
            throw std::logic_error(
                    "cosine similarity is not defined whenever one or both "
                    "input vectors are zero-vectors.");
        }

        return mul / (sqrt(d_a) * sqrt(d_b));*/

        double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
        for(unsigned int i = 0; i < size; ++i) {
            dot += A[i] * B[i] ;
            denom_a += A[i] * A[i] ;
            denom_b += B[i] * B[i] ;
        }

        return dot / (sqrt(denom_a) * sqrt(denom_b));
    }

    inline float equality(const float *A, const float *B, unsigned int size)
    {
        bool acc = true;
        for(unsigned int i = 0; i < size and acc == true; ++i) {
            acc = acc and (A[i]==B[i]);
        }

        if(acc) return 1.0f;
        else return 0.0f;
    }

    inline float SIMD_cosine_similarity(float *A, float *B, unsigned int size)
    {
        float distance;

        SimdCosineDistance32f(A, B, size, &distance);

        return 1 - distance;
    }

    inline float SIMD_cosine_similarity(const float *A, const float *B, unsigned int size)
    {
        float distance;

        SimdCosineDistance32f(A, B, size, &distance);

        return 1 - distance;
    }

    inline float SIMD_DotProduct(const float * a, const float * b, size_t size)
    {
        size_t i = 0, aligned = size&(~3);
        float sums[4] = { 0, 0, 0, 0 };
        for (; i < aligned; i += 4)
        {
            sums[0] += a[i + 0] * b[i + 0];
            sums[1] += a[i + 1] * b[i + 1];
            sums[2] += a[i + 2] * b[i + 2];
            sums[3] += a[i + 3] * b[i + 3];
        }
        for (; i < size; ++i)
            sums[0] += a[i] * b[i];
        return sums[0] + sums[1] + sums[2] + sums[3];
    }
};

#endif //J2V_JOIN2VEC_H
