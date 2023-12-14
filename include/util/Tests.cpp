//
// Created by sanca on 9/22/23.
//

#include "Tests.h"
#include "TimeBlock.h"

#include <random>
#include "Simd/SimdLib.h"

#include <atomic>
#include <mkl.h>

#include "Eigen/Dense"
#include "Eigen/Core"


Tests::Tests() {}

//#define SUM_ONLY
#undef SUM_ONLY

void Tests::scan_columns(size_t byte_width, size_t count, int iter) {
    cout<<"TEST SCAN COLUMN" << endl;

    vector<char*> column;
    long sum = 0;

    for(size_t i=0; i<count; i++){
        char* tmp = (char*)malloc(byte_width*sizeof(char));

        for(int j=0; j < byte_width; j++)
            tmp[j] = rand()%255+1;

        column.push_back(tmp);
    }

    {
        TimeBlock t(string("scan_columns_"+ to_string(byte_width)+ "_" +to_string(count)+"_"+to_string(iter)));
        for (auto &el: column) {
            sum += el[0];
        }
    }

    for(size_t i=0; i<count; i++){
        free(column[i]);
    }

    cout << sum << endl;

}

void Tests::scan_columns2(size_t count, int repeat, vector<int> offsets) {
    cout<<"TEST SCAN COLUMN2" << endl;

    long sum = 0;
    char* data;

    char* data_drop;

    cout << *max_element(offsets.begin(),offsets.end()) << endl;

    //data = (char*)malloc( count*(*max_element(offsets.begin(),offsets.end()))*sizeof(char));
    data = (char*) aligned_alloc(64, count*(*max_element(offsets.begin(),offsets.end()))*sizeof(char));
    data_drop = (char*)malloc( 1024*1024*2024*sizeof(char));

    cout << "count = " << count*(*max_element(offsets.begin(),offsets.end())) << endl;

    if(!data){
        cout << "NOT ALLOCATED!" << endl;
    } else {
        cout << "ALLOCATED!" << endl;
    }

    for(size_t j=0; j < count*(*max_element(offsets.begin(),offsets.end())); j++) {
        data[j] = rand() % 255 + 1;
    }

    for(size_t j=0; j < 1024*1024*2024; j++) {
        data_drop[j] = rand()%255 - 1;
    }


    cout << data[50] << ", " << data[1576] << endl;

    for(auto off: offsets) {
        for (int j = 0; j < repeat; j++) {
            system("dropcache");

            sum = 0;
            {
                TimeBlock t(
                        string("scan_columns2_" + to_string(count) + "_" + to_string(off)));

                for (size_t i = 0; i < count*off; i += off) {
                    sum += data[i];
                }
            }

            long sum_drop = 0;
            for (size_t i = 0; i < 1024*1024*2024; i++) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;

            sum_drop = 0;
            for (size_t i = 1024*1024*2024-1; i > 0 ; i--) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;

            sum_drop = 0;
            for (size_t i = 0; i < 1024*1024*2024-2; i+=2) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;

            sum_drop = 0;
            for (size_t i = 1024*1024*2024-1; i > 0 ; i--) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;
        }

    }

    free(data);
    free(data_drop);
}


void Tests::scan_columns_rand(size_t count, int repeat, vector<int> offsets) {
    cout<<"TEST SCAN COLUMN2" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    long sum = 0;
    char* data;

    char* data_drop;

    cout << *max_element(offsets.begin(),offsets.end()) << endl;

    //data = (char*)malloc( count*(*max_element(offsets.begin(),offsets.end()))*sizeof(char));
    data = (char*) aligned_alloc(64, count*(*max_element(offsets.begin(),offsets.end()))*sizeof(char));
    data_drop = (char*)malloc( 1024*1024*2024*sizeof(char));

    cout << "count = " << count*(*max_element(offsets.begin(),offsets.end())) << endl;

    if(!data){
        cout << "NOT ALLOCATED!" << endl;
    } else {
        cout << "ALLOCATED!" << endl;
    }

    for(size_t j=0; j < count*(*max_element(offsets.begin(),offsets.end())); j++) {
        data[j] = rand() % 255 + 1;
    }

    for(size_t j=0; j < 1024*1024*2024; j++) {
        data_drop[j] = rand()%255 - 1;
    }

    cout << data[50] << ", " << data[1576] << endl;

    for(auto off: offsets) {
        cout << "OFFSET = " << off << endl;
        array<size_t, 1000000> idx{};

        std::uniform_int_distribution<int> distribution(0,off);

        for(int i=0; i<1000000; i++){
            //idx[i] = i*off+rand()%off;
            idx[i] = i*off+distribution(generator);
        }

        for (int j = 0; j < repeat; j++) {
            system("dropcache");

            sum = 0;
            {
                TimeBlock t(
                        string("scan_columns_random_" + to_string(count) + "_" + to_string(off)));

                for (auto id: idx) {
                    sum += data[id];
                }
            }

            long sum_drop = 0;
            for (size_t i = 0; i < 1024*1024*2024; i++) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;

            sum_drop = 0;
            for (size_t i = 1024*1024*2024-1; i > 0 ; i--) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;

            sum_drop = 0;
            for (size_t i = 0; i < 1024*1024*2024-2; i+=2) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;

            sum_drop = 0;
            for (size_t i = 1024*1024*2024-1; i > 0 ; i--) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;
        }

    }

    free(data);
    free(data_drop);
}

void Tests::scan_columns_rand2(size_t count, int repeat, vector<int> offsets) {
    cout<<"TEST SCAN COLUMN2" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    long sum = 0;
    char* data;

    char* data_drop;

    cout << *max_element(offsets.begin(),offsets.end()) << endl;

    //data = (char*)malloc( count*(*max_element(offsets.begin(),offsets.end()))*sizeof(char));
    data = (char*) aligned_alloc(64, count*(*max_element(offsets.begin(),offsets.end()))*sizeof(char));
    data_drop = (char*)malloc( 1024*1024*2024*sizeof(char));

    cout << "count = " << count*(*max_element(offsets.begin(),offsets.end())) << endl;

    if(!data){
        cout << "NOT ALLOCATED!" << endl;
    } else {
        cout << "ALLOCATED!" << endl;
    }

    for(size_t j=0; j < count*(*max_element(offsets.begin(),offsets.end())); j++) {
        data[j] = rand() % 255 + 1;
    }

    for(size_t j=0; j < 1024*1024*2024; j++) {
        data_drop[j] = rand()%255 - 1;
    }

    cout << data[50] << ", " << data[1576] << endl;

    size_t len = count*(*max_element(offsets.begin(),offsets.end()));

    for(auto off: offsets) {
        cout << "OFFSET = " << off << endl;
        array<size_t, 1000000> idx{};

        std::uniform_int_distribution<int> distribution(std::max(0, off-64),off);

        for(int i=0; i<1000000; i++){
            idx[i] = i*off+distribution(generator);
        }

        for (int j = 0; j < repeat; j++) {
            system("dropcache");

            sum = 0;
            {
                TimeBlock t(
                        string("scan_columns_random_" + to_string(count) + "_" + to_string(off)));

                for (auto id: idx) {
                    sum += data[id%len];
                    data[id%len]++;
                }
            }

            long sum_drop = 0;
            for (size_t i = 0; i < 1024*1024*2024; i++) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;

            sum_drop = 0;
            for (size_t i = 1024*1024*2024-1; i > 0 ; i--) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;

            sum_drop = 0;
            for (size_t i = 0; i < 1024*1024*2024-2; i+=2) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;

            sum_drop = 0;
            for (size_t i = 1024*1024*2024-1; i > 0 ; i--) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;
        }

    }

    free(data);
    free(data_drop);
}


void Tests::scan_columns_rand3(size_t count, int repeat, vector<int> offsets) {
    cout<<"TEST SCAN COLUMN2" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    long sum = 0;
    char* data;

    char* data_drop;

    cout << *max_element(offsets.begin(),offsets.end()) << endl;

    //data = (char*)malloc( count*(*max_element(offsets.begin(),offsets.end()))*sizeof(char));
    data = (char*) aligned_alloc(64, count*(std::min(*max_element(offsets.begin(),offsets.end()), 2048))*sizeof(char));
    data_drop = (char*)malloc( 1024*1024*2024*sizeof(char));

    cout << "count = " << count*(*max_element(offsets.begin(),offsets.end())) << endl;

    if(!data){
        cout << "NOT ALLOCATED!" << endl;
    } else {
        cout << "ALLOCATED!" << endl;
    }

    for(size_t j=0; j < count*(std::min(*max_element(offsets.begin(),offsets.end()), 2048)); j++) {
        data[j] = rand() % 255 + 1;
    }

    for(size_t j=0; j < 1024*1024*2024; j++) {
        data_drop[j] = rand()%255 - 1;
    }

    cout << data[50] << ", " << data[1576] << endl;

    size_t len = count*(std::min(*max_element(offsets.begin(),offsets.end()), 2048));

    for(auto off: offsets) {
        cout << "OFFSET = " << off << endl;
        array<size_t, 1000000> idx{};

        std::uniform_int_distribution<int> distribution(std::max(0, off-64),off);

        for(int i=0; i<1000000; i++){
            idx[i] = i*off+distribution(generator);
        }

        for (int j = 0; j < repeat; j++) {
            system("dropcache");

            sum = 0;
            {
                TimeBlock t(
                        string("scan_columns_random_" + to_string(count) + "_" + to_string(off)));

                for (auto id: idx) {
                    sum += data[id%len];
                    data[id%len]++;
                }
            }

            long sum_drop = 0;
            for (size_t i = 0; i < 1024*1024*2024; i++) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;

            sum_drop = 0;
            for (size_t i = 1024*1024*2024-1; i > 0 ; i--) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;

            sum_drop = 0;
            for (size_t i = 0; i < 1024*1024*2024-2; i+=2) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;

            sum_drop = 0;
            for (size_t i = 1024*1024*2024-1; i > 0 ; i--) {
                sum_drop += data_drop[i];
            }

            system("dropcache");
            cout << sum << endl;
            cout << sum_drop << endl;
        }

    }

    free(data);
    free(data_drop);
}

void Tests::scan_vector(size_t byte_width, size_t vector_size, size_t count) {
    cout<<"TEST SCAN VECTOR" << endl;
}

void Tests::scan_vector_columns() {

}

inline float SIMD_cosine_similarity(float *A, float *B, unsigned int size)
{
    float distance;

    SimdCosineDistance32f(A, B, size, &distance);

    return 1 - distance;
}

inline float cosine_similarity(const float *A, const float *B, unsigned int size)
{
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for(unsigned int i = 0; i < size; ++i) {
        dot += A[i] * B[i] ;
        denom_a += A[i] * A[i] ;
        denom_b += B[i] * B[i] ;
    }

    return dot / (sqrt(denom_a) * sqrt(denom_b));
}

inline float cosine_similarity(const double *A, const double *B, unsigned int size)
{
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for(unsigned int i = 0; i < size; ++i) {
        dot += A[i] * B[i] ;
        denom_a += A[i] * A[i] ;
        denom_b += B[i] * B[i] ;
    }

    return dot / (sqrt(denom_a) * sqrt(denom_b));
}

void Tests::cache_cosine_sim_vec(size_t arrayA, size_t vectorD, float th, bool SIMD){

    cout<<"COSINE TEST" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    //generate vectors
    float* in = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));
    float* comp = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));

    //float* in = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));
    //float* comp = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));

    for(size_t i=0; i< arrayA * vectorD; i++){
        in[i] = distribution(generator);
    }

    for(size_t i=0; i<  arrayA * vectorD; i++){
        comp[i] = distribution(generator);
    }

    float sim = 0;
    long cnt = 0;

    for(int iter=0; iter<5; iter++)
    {
        TimeBlock t(string("cosine_cache_vec_" + to_string(arrayA) + "_" + to_string(vectorD)));
        for (size_t i = 0; i < arrayA*vectorD; i+=vectorD) {
            for (size_t j = 0; j < arrayA*vectorD; j+=vectorD) {
#ifdef SUM_ONLY
                sim += SIMD_cosine_similarity((in + i), (comp + j), vectorD); //SIMD_cosine_similarity((in + i), (comp + j), vectorD);
#else
                if(SIMD_cosine_similarity((in + i), (comp + j), vectorD) >= th)
                    cnt++;
#endif

            }
        }
    }

    cout << sim << endl;
    cout << cnt << endl;

    this->resCnt[string("cosine_cache_vec-" + to_string(arrayA) + "-" + to_string(vectorD))].push_back(cnt);

    free(in);
    free(comp);
}


void Tests::cache_cosine_sim(size_t arrayA, size_t vectorD, float th, bool SIMD){

    cout<<"COSINE TEST" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    //generate vectors
    float* in = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));
    float* comp = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));

    //float* in = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));
    //float* comp = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));

    for(size_t i=0; i< arrayA * vectorD; i++){
        in[i] = distribution(generator);
    }

    for(size_t i=0; i<  arrayA * vectorD; i++){
        comp[i] = distribution(generator);
    }

    float sim = 0;
    long cnt = 0;

    for(int iter=0; iter<5; iter++)
    {
        TimeBlock t(string("cosine_cache_sim_" + to_string(arrayA) + "_" + to_string(vectorD)));
        for (size_t i = 0; i < arrayA*vectorD; i+=vectorD) {
            for (size_t j = 0; j < arrayA*vectorD; j+=vectorD) {
#ifdef SUM_ONLY
                sim += cosine_similarity((in + i), (comp + j), vectorD); //SIMD_cosine_similarity((in + i), (comp + j), vectorD);
#else
                if(cosine_similarity((in + i), (comp + j), vectorD) >= th)
                    cnt++;
#endif

            }
        }
    }

    cout << sim << endl;
    cout << cnt << endl;
    this->resCnt[string("cosine_cache_sim-" + to_string(arrayA) + "-" + to_string(vectorD))].push_back(cnt);

    free(in);
    free(comp);
}


void Tests::cache_cosine_sim64(size_t arrayA, size_t vectorD, float th, bool SIMD){

    cout<<"COSINE TEST" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<double> distribution(0.0f, 1.0f);

    //generate vectors
    double* in = static_cast<double *>(aligned_alloc(64, arrayA * vectorD * sizeof(double)));
    double* comp = static_cast<double *>(aligned_alloc(64, arrayA * vectorD * sizeof(double)));

    //float* in = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));
    //float* comp = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));

    for(size_t i=0; i< arrayA * vectorD; i++){
        in[i] = distribution(generator);
    }

    for(size_t i=0; i<  arrayA * vectorD; i++){
        comp[i] = distribution(generator);
    }

    float sim = 0;
    long cnt = 0;

    for(int iter=0; iter<5; iter++)
    {
        TimeBlock t(string("cosine_cache_sim_" + to_string(arrayA) + "_" + to_string(vectorD)));
        for (size_t i = 0; i < arrayA*vectorD; i+=vectorD) {
            for (size_t j = 0; j < arrayA*vectorD; j+=vectorD) {
#ifdef SUM_ONLY
                sim += cosine_similarity((in + i), (comp + j), vectorD); //SIMD_cosine_similarity((in + i), (comp + j), vectorD);
#else
                if(cosine_similarity((in + i), (comp + j), vectorD) >= th)
                    cnt++;
#endif

            }
        }
    }

    cout << sim << endl;
    cout << cnt << endl;
    this->resCnt[string("cosine_cache_sim-" + to_string(arrayA) + "-" + to_string(vectorD))].push_back(cnt);

    free(in);
    free(comp);
}


void Tests::cache_cosine_sim1(size_t arrayA, size_t vectorD, float th, bool SIMD){

    cout<<"COSINE TEST" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    //generate vectors
    float* in = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));
    float* comp = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));

    //float* in = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));
    //float* comp = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));

    for(size_t i=0; i< arrayA * vectorD; i++){
        in[i] = distribution(generator);
    }

    for(size_t i=0; i<  arrayA * vectorD; i++){
        comp[i] = distribution(generator);
    }

    float sim = 0;
    long cnt = 0;
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0;

    for(int iter=0; iter<5; iter++)
    {
        TimeBlock t(string("cosine_cache_sim1_" + to_string(arrayA) + "_" + to_string(vectorD)));
        for (size_t i = 0; i < arrayA*vectorD; i+=vectorD) {
            for (size_t j = 0; j < arrayA*vectorD; j+=vectorD) {

                dot = 0.0;
                denom_a = 0.0;
                denom_b = 0.0;

                for(unsigned int k = 0; k < vectorD; ++k) {
                    dot += (in+i)[k] * (comp+j)[k] ;
                    denom_a += (in+i)[k] * (comp+j)[k] ;
                    denom_b += (in+i)[k] * (comp+j)[k] ;
                }

#ifdef SUM_ONLY
                sim += dot / (sqrt(denom_a) * sqrt(denom_b));
#else
                if(dot / (sqrt(denom_a) * sqrt(denom_b)) >= th)
                    cnt++;
#endif

                //sim += cosine_similarity((in + i), (comp + j), vectorD); //SIMD_cosine_similarity((in + i), (comp + j), vectorD);
            }
        }
    }

    cout << sim << endl;
    cout << cnt << endl;

    this->resCnt[string("cosine_cache_sim1-" + to_string(arrayA) + "-" + to_string(vectorD))].push_back(cnt);

    free(in);
    free(comp);
}

void Tests::cache_cosine_sim11(size_t arrayA, float th, bool SIMD){
    size_t vectorD = 1;
    cout<<"COSINE TEST" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    //generate vectors
    float* in = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));
    float* comp = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));

    //float* in = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));
    //float* comp = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));

    for(size_t i=0; i< arrayA * vectorD; i++){
        in[i] = distribution(generator);
    }

    for(size_t i=0; i<  arrayA * vectorD; i++){
        comp[i] = distribution(generator);
    }

    float sim = 0;
    long cnt = 0;
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0;

    for(int iter=0; iter<5; iter++)
    {
        TimeBlock t(string("cosine_cache_" + to_string(arrayA) + "_" + to_string(vectorD)));
        for (size_t i = 0; i < arrayA*vectorD; i+=vectorD) {
            for (size_t j = 0; j < arrayA*vectorD; j+=vectorD) {

                dot = *(in+i) * *(comp+j);
                denom_a = *(in+i) * *(comp+j);
                denom_b = *(in+i) * *(comp+j);

#ifdef SUM_ONLY
                sim += dot / (sqrt(denom_a) * sqrt(denom_b));
#else
                if(dot / (sqrt(denom_a) * sqrt(denom_b)) >= th)
                    cnt++;
#endif

                //sim += cosine_similarity((in + i), (comp + j), vectorD); //SIMD_cosine_similarity((in + i), (comp + j), vectorD);
            }
        }
    }

    cout << sim << endl;
    cout << cnt << endl;

    this->resCnt[string("cosine_cache_" + to_string(arrayA) + "_" + to_string(vectorD))].push_back(cnt);

    free(in);
    free(comp);
}


inline void normalize_self(float *A, size_t a, size_t d){

    size_t off;

    float vec_l2;
    for(size_t i = 0; i<a; i++) {
        vec_l2 = cblas_snrm2(d, A + (d * i), 1);  // stride is 1, normalize over d, repeat this per-vector
        cblas_sscal(d, 1.0 / vec_l2, A + (d * i), 1);
    }

}

inline void normalize_self(double *A, size_t a, size_t d){

    size_t off;

    double vec_l2;
    for(size_t i = 0; i<a; i++) {
        vec_l2 = cblas_dnrm2(d, A + (d * i), 1);  // stride is 1, normalize over d, repeat this per-vector
        cblas_dscal(d, 1.0 / vec_l2, A + (d * i), 1);
    }

}

void Tests::cache_cosine_sim_blas(size_t arrayA, size_t vectorD, size_t batch, float th){

    cout<<"COSINE TEST BLAS" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    //generate vectors
    float* in = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));
    float* comp = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));

    for(size_t i=0; i< arrayA * vectorD; i++){
        in[i] = distribution(generator);
    }

    for(size_t i=0; i<  arrayA * vectorD; i++){
        comp[i] = distribution(generator);
    }


    cout << "CPU BLAS JOIN START" << endl;

    size_t D = vectorD, R = arrayA, S = arrayA;
    std::vector<float> results;
    results.resize(R*S);

    {
        TimeBlock tr(string("normR_" + to_string(arrayA) + "_" + to_string(vectorD)));
        normalize_self(in, R, D);
    }

    {
        TimeBlock ts(string("normS_" + to_string(arrayA) + "_" + to_string(vectorD)));
        normalize_self(comp, S, D);
    }

    float sim = 0.0;
    long cnt = 0;

    for(int iter=0; iter<5; iter++) {
        /**
         * Rv = R x D
         * Sv = S x D   -- needs to be transposed, effectively D x S - check if ldb is correct (S or D)
         * res = R x S
         */
        cout << "GEMM START" << endl;


        {
            TimeBlock t_gemm(string("gemm_" + to_string(arrayA) + "_" + to_string(vectorD)));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, R, S, D, 1.0, in, D, comp, D, 0.0,
                        results.data(), S);
            //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, R, S, D, 1.0, R_norm, D, S_norm, D, 0.0, results.data(), S);

#ifdef SUM_ONLY
            // TODO: check with the summation!
            for(size_t t=0; t<results.size(); t++){
                sim+=results[t];
            }
#else
            for(size_t t=0; t<results.size(); t++){
                if(results[t]>=th)
                    cnt++;
            }
#endif
        }
        cout << "GEMM END" << endl;
    }

    cout << sim << endl;
    cout << cnt << endl;
    this->resCnt[string("gemm-" + to_string(arrayA) + "-" + to_string(vectorD))].push_back(cnt);

    free(in);
    free(comp);

//    {
//        TimeBlock t_res("get results");
//        std::vector<std::future<vector<res_pair>>> final_results;
//
//        size_t parallelism = tp.getSize();
//        size_t step = ceil(R / (double) parallelism);
//
//        for (size_t t = 0; t < parallelism; t++) {
//            size_t begin = t * step;
//            size_t end;
//            if (t == parallelism - 1) {
//                end = R;
//            } else {
//                end = (t + 1) * step;
//            }
//
//            final_results.emplace_back(
//                    tp.enqueue([this, begin, end, threshold, S, &results]() {
//                        vector<res_pair> res;
//                        size_t off;
//
//                        //float max=-2000, min=2000;
//
//                        for (size_t i = begin; i < end; i++) {
//                            off = i * S;
//                            for (size_t j = 0; j < S; j++) {
//                                if (results[off + j] >= threshold) {
//                                    res.push_back({i, j});
//                                }
//
//                                //if(results[off+j]>max) max = results[off+j];
//                                //if(results[off+j]<min) min = results[off+j];
//                            }
//                        }
//
//                        //cout << "MIN = " << min << ", MAX = " << max << endl;
//
//                        return res;
//                    })
//            );
//        }

}



void Tests::cache_cosine_sim_blas1(size_t arrayA, size_t vectorD, size_t batch, float th){

    cout<<"COSINE TEST BLAS 1" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    //generate vectors
    float* in = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));
    float* comp = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));

    for(size_t i=0; i< arrayA * vectorD; i++){
        in[i] = distribution(generator);
    }

    for(size_t i=0; i<  arrayA * vectorD; i++){
        comp[i] = distribution(generator);
    }


    cout << "CPU BLAS JOIN START" << endl;

    size_t D = vectorD, R = arrayA, S = arrayA;
    std::vector<float> results;
    results.resize(R*S);

    {
        TimeBlock tr(string("normR1_" + to_string(arrayA) + "_" + to_string(vectorD)));
        normalize_self(in, R, D);
    }

    {
        TimeBlock ts(string("normS1_" + to_string(arrayA) + "_" + to_string(vectorD)));
        for(long i=0; i<S; i++) {
            normalize_self(&comp[i*D], 1, D);
        }
    }

    float sim = 0.0;
    long cnt = 0;

    for(int iter=0; iter<5; iter++) {
        /**
         * Rv = R x D
         * Sv = S x D   -- needs to be transposed, effectively D x S - check if ldb is correct (S or D)
         * res = R x S
         */
        cout << "GEMM START" << endl;


        {
            TimeBlock t_gemm(string("gemm1_" + to_string(arrayA) + "_" + to_string(vectorD)));
            for(long i=0; i< S; i++) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, R, 1, D, 1.0, in, D, &comp[i], D, 0.0,
                            &results.data()[i], 1);
                //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, R, S, D, 1.0, R_norm, D, S_norm, D, 0.0, results.data(), S);

#ifdef SUM_ONLY
                // TODO: check with the summation!
                for(size_t t=0; t<results.size(); t++){
                    sim+=results[t];
                }
#else
                //for (size_t t = 0; t < results.size(); t++) {
                    if (results[i] >= th)
                        cnt++;
                //}
#endif
            }
        }
        cout << "GEMM END" << endl;
    }

    cout << sim << endl;
    cout << cnt << endl;
    this->resCnt[string("gemm1-" + to_string(arrayA) + "-" + to_string(vectorD))].push_back(cnt);

    free(in);
    free(comp);
}


void Tests::cache_cosine_sim_blas64(size_t arrayA, size_t vectorD, size_t batch, float th){

    cout<<"COSINE TEST BLAS" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<double> distribution(0.0f, 1.0f);

    //generate vectors
    double* in = static_cast<double *>(aligned_alloc(64, arrayA * vectorD * sizeof(double)));
    double* comp = static_cast<double *>(aligned_alloc(64, arrayA * vectorD * sizeof(double)));

    for(size_t i=0; i< arrayA * vectorD; i++){
        in[i] = distribution(generator);
    }

    for(size_t i=0; i<  arrayA * vectorD; i++){
        comp[i] = distribution(generator);
    }


    cout << "CPU BLAS JOIN START" << endl;

    size_t D = vectorD, R = arrayA, S = arrayA;
    std::vector<double> results;
    results.resize(R*S);

    {
        TimeBlock tr(string("normR_" + to_string(arrayA) + "_" + to_string(vectorD)));
        normalize_self(in, R, D);
    }

    {
        TimeBlock ts(string("normS_" + to_string(arrayA) + "_" + to_string(vectorD)));
        normalize_self(comp, S, D);
    }

    float sim = 0.0;
    long cnt = 0;

    for(int iter=0; iter<5; iter++) {
        /**
         * Rv = R x D
         * Sv = S x D   -- needs to be transposed, effectively D x S - check if ldb is correct (S or D)
         * res = R x S
         */
        cout << "GEMM START" << endl;


        {
            TimeBlock t_gemm(string("gemm_" + to_string(arrayA) + "_" + to_string(vectorD)));
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, R, S, D, 1.0, in, D, comp, D, 0.0,
                        results.data(), S);
            //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, R, S, D, 1.0, R_norm, D, S_norm, D, 0.0, results.data(), S);

#ifdef SUM_ONLY
            // TODO: check with the summation!
            for(size_t t=0; t<results.size(); t++){
                sim+=results[t];
            }
#else
            for(size_t t=0; t<results.size(); t++){
                if(results[t]>=th)
                    cnt++;
            }
#endif
        }
        cout << "GEMM END" << endl;
    }

    cout << sim << endl;
    cout << cnt << endl;
    this->resCnt[string("gemm-" + to_string(arrayA) + "-" + to_string(vectorD))].push_back(cnt);

    free(in);
    free(comp);
}

void Tests::cache_cosine_sim_blas_normalize_only(size_t arrayA, size_t vectorD, size_t batch, float th) {

    cout << "COSINE TEST BLAS NORMALIZE ONLY" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    //generate vectors
    float *in = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));
    float *comp = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));

    for (size_t i = 0; i < arrayA * vectorD; i++) {
        in[i] = distribution(generator);
    }

    for (size_t i = 0; i < arrayA * vectorD; i++) {
        comp[i] = distribution(generator);
    }

    size_t D = vectorD, R = arrayA, S = arrayA;
    std::vector<float> results;
    results.resize(R * S);

    for(int i=0; i<5; i++) {
        {
            TimeBlock tr(string("normR_" + to_string(arrayA) + "_" + to_string(vectorD)));
            normalize_self(in, R, D);
        }

        {
            TimeBlock ts(string("normS_" + to_string(arrayA) + "_" + to_string(vectorD)));
            normalize_self(comp, S, D);
        }
    }

    free(in);
    free(comp);
}


void Tests::cache_cosine_sim_blas_normalize_only1(size_t arrayA, size_t vectorD, size_t batch, float th) {

    cout << "COSINE TEST BLAS NORMALIZE ONLY1" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    //generate vectors
    float *in = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));
    float *comp = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));

    for (size_t i = 0; i < arrayA * vectorD; i++) {
        in[i] = distribution(generator);
    }

    for (size_t i = 0; i < arrayA * vectorD; i++) {
        comp[i] = distribution(generator);
    }

    size_t D = vectorD, R = arrayA, S = arrayA;
    std::vector<float> results;
    results.resize(R * S);

    for(int i=0; i<5; i++) {
        {
            TimeBlock tr(string("normR_" + to_string(arrayA) + "_" + to_string(vectorD)));
            normalize_self(in, R, D);
        }

        {
            TimeBlock ts(string("normS1_" + to_string(arrayA) + "_" + to_string(vectorD)));
            for(long i=0; i<S; i++) {
                normalize_self(&comp[i*D], 1, D);
            }
        }
    }

    free(in);
    free(comp);
}

void Tests::cache_cosine_sim_blas_normalize_only64(size_t arrayA, size_t vectorD, size_t batch, float th) {

    cout << "COSINE TEST BLAS NORMALIZE ONLY" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<double> distribution(0.0f, 1.0f);

    //generate vectors
    double *in = static_cast<double *>(aligned_alloc(64, arrayA * vectorD * sizeof(double)));
    double *comp = static_cast<double *>(aligned_alloc(64, arrayA * vectorD * sizeof(double)));

    for (size_t i = 0; i < arrayA * vectorD; i++) {
        in[i] = distribution(generator);
    }

    for (size_t i = 0; i < arrayA * vectorD; i++) {
        comp[i] = distribution(generator);
    }

    size_t D = vectorD, R = arrayA, S = arrayA;
    std::vector<double> results;
    results.resize(R * S);

    for(int i=0; i<5; i++) {
        {
            TimeBlock tr(string("normR_" + to_string(arrayA) + "_" + to_string(vectorD)));
            normalize_self(in, R, D);
        }

        {
            TimeBlock ts(string("normS_" + to_string(arrayA) + "_" + to_string(vectorD)));
            normalize_self(comp, S, D);
        }
    }

    free(in);
    free(comp);
}

void Tests::blas_mult_only(size_t arrayA, size_t arrayB, size_t vectorD, size_t batch){
    /*cout<<"COSINE TEST" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    //generate vectors
    float* in = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));
    float* comp = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));

    //float* in = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));
    //float* comp = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));

    for(size_t i=0; i< arrayA * vectorD; i++){
        in[i] = distribution(generator);
    }

    for(size_t i=0; i<  arrayA * vectorD; i++){
        comp[i] = distribution(generator);
    }

    float sim = 0;
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0;

    for(int iter=0; iter<5; iter++)
    {
        TimeBlock t(string("cosine_cache_" + to_string(arrayA) + "_" + to_string(vectorD)));
        for (size_t i = 0; i < arrayA*vectorD; i+=vectorD) {
            for (size_t j = 0; j < arrayA*vectorD; j+=vectorD) {

                dot = *(in+i) * *(comp+j);
                denom_a = *(in+i) * *(comp+j);
                denom_b = *(in+i) * *(comp+j);

                sim += dot / (sqrt(denom_a) * sqrt(denom_b));

                //sim += cosine_similarity((in + i), (comp + j), vectorD); //SIMD_cosine_similarity((in + i), (comp + j), vectorD);
            }
        }
    }

    cout << sim << endl;*/
}

void Tests::cache_cosine_sim_block1(size_t arrayA, bool SIMD){
    size_t vectorD = 1;
    cout<<"COSINE TEST" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    //generate vectors
    //float* in = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));
    //float* comp = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));

    float* in = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));
    float* comp = static_cast<float *>(malloc( arrayA * vectorD * sizeof(float)));

    for(size_t i=0; i< arrayA * vectorD; i++){
        in[i] = distribution(generator);
    }

    for(size_t i=0; i<  arrayA * vectorD; i++){
        comp[i] = distribution(generator);
    }

    float sim = 0;

    for(int iter=0; iter<5; iter++)
    {
        TimeBlock t(string("cosine_cache_" + to_string(arrayA) + "_" + to_string(vectorD)));
        for (size_t i = 0; i < arrayA; i++) {
            for (size_t j = 0; j < arrayA; j+=4) {
                sim += cosine_similarity((in + i), (comp + j), vectorD); //SIMD_cosine_similarity((in + i), (comp + j), vectorD);
                sim += cosine_similarity((in + i), (comp + j+1), vectorD);
                sim += cosine_similarity((in + i), (comp + j+2), vectorD);
                sim += cosine_similarity((in + i), (comp + j+3), vectorD);
            }
        }
    }

    cout << sim << endl;
}


void Tests::join_BLAS(size_t arrayA, size_t arrayB, size_t vectorD, float thr){

    cout<<"COSINE TEST BLAS JOIN" << endl;

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    //generate vectors
    float* in = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));
    float* comp = static_cast<float *>(aligned_alloc(64, arrayB * vectorD * sizeof(float)));

    for(size_t i=0; i< arrayA * vectorD; i++){
        in[i] = distribution(generator);
    }

    for(size_t i=0; i<  arrayB * vectorD; i++){
        comp[i] = distribution(generator);
    }


    cout << "CPU BLAS JOIN START" << endl;

    size_t D = vectorD, R = arrayA, S = arrayB;
    //std::vector<float> results;

    //results.resize(R*S);

    /**
     * EVEN CPU HAS A LIMITED MEMORY - BATCH/BLOCK
     */

    float* results = static_cast<float *>(aligned_alloc(64, R * S * sizeof(float)));

    {
        TimeBlock tr(string("normR_" + to_string(arrayA) + "x" + to_string(arrayB) + "_" + to_string(vectorD)));
        normalize_self(in, R, D);
    }

    {
        TimeBlock ts(string("normS_" + to_string(arrayA) + "x" + to_string(arrayB) + "_" + to_string(vectorD)));
        normalize_self(comp, S, D);
    }

    float sim = 0.0;
    long cnt = 0;

    for(int iter=0; iter<5; iter++) {
        /**
         * Rv = R x D
         * Sv = S x D   -- needs to be transposed, effectively D x S - check if ldb is correct (S or D)
         * res = R x S
         */
        cout << "GEMM START" << endl;


        {
            TimeBlock t_gemm("gemm_" + to_string(arrayA) + "x" + to_string(arrayB) + "_" + to_string(vectorD));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, R, S, D, 1.0, in, D, comp, D, 0.0,
                        results, S);
            //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, R, S, D, 1.0, R_norm, D, S_norm, D, 0.0, results.data(), S);
        }
        cout << "GEMM END" << endl;
    }


    auto& tp = ThreadPool::getInstance();
    std::vector<std::future<vector<res_pair>>> final_results;

    {
        TimeBlock t_res(string("getRes_" + to_string(arrayA) + "x" + to_string(arrayB) + "_" + to_string(vectorD)));

        size_t parallelism = tp.getSize();
        size_t step = ceil(R / (double) parallelism);

        for (size_t t = 0; t < parallelism; t++) {
            size_t begin = t * step;
            size_t end;
            if (t == parallelism - 1) {
                end = R;
            } else {
                end = (t + 1) * step;
            }

            final_results.emplace_back(
                    tp.enqueue([this, begin, end, thr, S, &results]() {
                        vector<res_pair> res;
                        size_t off;

                        //float max=-2000, min=2000;

                        for (size_t i = begin; i < end; i++) {
                            off = i * S;
                            for (size_t j = 0; j < S; j++) {
                                if (results[off + j] >= thr) {
                                    res.push_back({i, j});
                                }

                                //if(results[off+j]>max) max = results[off+j];
                                //if(results[off+j]<min) min = results[off+j];
                            }
                        }

                        //cout << "MIN = " << min << ", MAX = " << max << endl;

                        return res;
                    })
            );
        }
    }

    vector<res_pair> ret;

    for(auto && res : final_results){
        const vector<res_pair>& tmp = res.get();
        ret.insert(ret.end(), tmp.begin(), tmp.end());
    }

    cout<<ret.size()<<endl;

    cout << sim << endl;
    cout << cnt << endl;

    free(in);
    free(comp);
}

void Tests::join_BLAS_batch(size_t arrayA, size_t arrayB, size_t vectorD, float thr, pair<size_t, size_t> batch){

    cout<<"COSINE TEST BLAS JOIN" << endl;

    bool isBatched = false;

    if(batch.first <= arrayA and batch.second <= arrayB){
        cout << "BATCH" << endl;
        isBatched = true;
    } else {
        cout << "NO BATCH" << endl;
    }

    std::default_random_engine generator;
    generator.seed(424242);

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    //generate vectors
    float* in = static_cast<float *>(aligned_alloc(64, arrayA * vectorD * sizeof(float)));
    float* comp = static_cast<float *>(aligned_alloc(64, arrayB * vectorD * sizeof(float)));

    for(size_t i=0; i< arrayA * vectorD; i++){
        in[i] = distribution(generator);
    }

    for(size_t i=0; i<  arrayB * vectorD; i++){
        comp[i] = distribution(generator);
    }

    cout << "CPU BLAS JOIN START" << endl;
    size_t D = vectorD, R = arrayA, S = arrayB;

    if(!isBatched) {

        float *results = static_cast<float *>(aligned_alloc(64, R * S * sizeof(float)));

        {
            TimeBlock tr(string("normR_" + to_string(arrayA) + "x" + to_string(arrayB) + "_" + to_string(vectorD)));
            normalize_self(in, R, D);
        }

        {
            TimeBlock ts(string("normS_" + to_string(arrayA) + "x" + to_string(arrayB) + "_" + to_string(vectorD)));
            normalize_self(comp, S, D);
        }

        float sim = 0.0;
        long cnt = 0;

        for (int iter = 0; iter < 5; iter++) {
            /**
             * Rv = R x D
             * Sv = S x D   -- needs to be transposed, effectively D x S - check if ldb is correct (S or D)
             * res = R x S
             */
            cout << "GEMM START" << endl;


            {
                TimeBlock t_gemm("gemm_" + to_string(arrayA) + "x" + to_string(arrayB) + "_" + to_string(vectorD));
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, R, S, D, 1.0, in, D, comp, D, 0.0,
                            results, S);
                //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, R, S, D, 1.0, R_norm, D, S_norm, D, 0.0, results.data(), S);
            }
            cout << "GEMM END" << endl;
        }


        auto &tp = ThreadPool::getInstance();
        std::vector<std::future<vector<res_pair>>> final_results;

        {
            TimeBlock t_res(string("getRes_" + to_string(arrayA) + "x" + to_string(arrayB) + "_" + to_string(vectorD)));

            size_t parallelism = tp.getSize();
            size_t step = ceil(R / (double) parallelism);

            for (size_t t = 0; t < parallelism; t++) {
                size_t begin = t * step;
                size_t end;
                if (t == parallelism - 1) {
                    end = R;
                } else {
                    end = (t + 1) * step;
                }

                final_results.emplace_back(
                        tp.enqueue([this, begin, end, thr, S, &results]() {
                            vector<res_pair> res;
                            size_t off;

                            //float max=-2000, min=2000;

                            for (size_t i = begin; i < end; i++) {
                                off = i * S;
                                for (size_t j = 0; j < S; j++) {
                                    if (results[off + j] >= thr) {
                                        res.push_back({i, j});
                                    }

                                    //if(results[off+j]>max) max = results[off+j];
                                    //if(results[off+j]<min) min = results[off+j];
                                }
                            }

                            //cout << "MIN = " << min << ", MAX = " << max << endl;

                            return res;
                        })
                );
            }
        }

        vector<res_pair> ret;

        for (auto &&res: final_results) {
            const vector<res_pair> &tmp = res.get();
            ret.insert(ret.end(), tmp.begin(), tmp.end());
        }

        cout << ret.size() << endl;

        cout << sim << endl;
        cout << cnt << endl;
    } else {

        cout << "BATCHED JOIN" << endl;
        {
            TimeBlock tr(string("normR_" + to_string(arrayA) + "x" + to_string(arrayB) + "_" + to_string(vectorD)));
            normalize_self(in, R, D);
        }

        {
            TimeBlock ts(string("normS_" + to_string(arrayA) + "x" + to_string(arrayB) + "_" + to_string(vectorD)));
            normalize_self(comp, S, D);
        }

        float sim = 0.0;
        long cnt = 0;


        size_t arrayABatches = ceil(arrayA / (double) batch.first);
        size_t arrayBBatches = ceil(arrayB / (double) batch.second);

        float *results = static_cast<float *>(aligned_alloc(64, batch.first * batch.second * sizeof(float)));

        int num_batches = 0;

        for (int iter = 0; iter < 5; iter++) {
            /**
             * Rv = R x D
             * Sv = S x D   -- needs to be transposed, effectively D x S - check if ldb is correct (S or D)
             * res = R x S
             */
            /**
             * in the batched case, we get smaller intermediate results that we evaluate
             */
            cout << "GEMM BATCHED START" << endl;
            auto &tp = ThreadPool::getInstance();
            std::vector<std::future<vector<res_pair>>> final_results;

            {
                TimeBlock t_gemm("gemm_" + to_string(arrayA) + "x" + to_string(arrayB) + "_" + to_string(vectorD)+ "_batched:" + to_string(batch.first) + "x" + to_string(batch.second));
                for(int stepA = 0; stepA < arrayA; stepA+=batch.first){
                    for(int stepB = 0; stepB < arrayB; stepB+=batch.second){

                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch.first, batch.second, D, 1.0, &in[stepA], D, &comp[stepB], D, 0.0,
                                    results, batch.second);

                        cout << "NB=" << num_batches << endl;
                        num_batches++;

                        {
                            TimeBlock t_res(
                                    string("getRes_" + to_string(arrayA) + "x" + to_string(arrayB) + "_" + to_string(vectorD)));

                            size_t parallelism = tp.getSize();
                            size_t step = ceil(batch.first / (double) parallelism);

                            for (size_t t = 0; t < parallelism; t++) {
                                size_t begin = t * step;
                                size_t end;
                                if (t == parallelism - 1) {
                                    end = batch.first;
                                } else {
                                    end = (t + 1) * step;
                                }

                                final_results.emplace_back(
                                        tp.enqueue([begin, end, thr, batch, &results]() {
                                            vector<res_pair> res;
                                            size_t off;

                                            //float max=-2000, min=2000;

                                            for (size_t i = begin; i < end; i++) {
                                                off = i * batch.second;
                                                for (size_t j = 0; j < batch.second; j++) {
                                                    if (results[off + j] >= thr) {
                                                        res.push_back({i, j});
                                                    }

                                                    //if(results[off+j]>max) max = results[off+j];
                                                    //if(results[off+j]<min) min = results[off+j];
                                                }
                                            }

                                            //cout << "MIN = " << min << ", MAX = " << max << endl;

                                            return res;
                                        })
                                );
                            }
                        }

                        vector<res_pair> ret;

                        for (auto &&res: final_results) {
                            const vector<res_pair> &tmp = res.get();
                            ret.insert(ret.end(), tmp.begin(), tmp.end());
                        }
                        cout << ret.size() << endl;
                        final_results.clear();
                    }
                }
            }

            num_batches = 0;
            cout << "GEMM END" << endl;
            cout << "NUM BATCHES = " << num_batches << endl;

            cout << sim << endl;
            cout << cnt << endl;

        }
    }
    free(in);
    free(comp);
}