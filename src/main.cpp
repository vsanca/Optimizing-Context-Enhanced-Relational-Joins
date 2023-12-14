//
// Created by Viktor Sanca on 9/26/22.
//

#include <iostream>
#include <map>

#include "fasttext/args.h"
#include "fasttext/autotune.h"
#include "fasttext/fasttext.h"

#include "join2vec/join2vec.h"
#include "join2vec/hashjoin2vec.h"
#include "join2vec/scan.h"

#include "util/ThreadPool.h"
#include "util/TimeBlock.h"
#include "join2vec/LSH.h"
#include "cluster.h"

#include <chrono>

#include "Eigen/Dense"
#include "Eigen/Core"

#include "util/Tests.h"

#include <chrono>
#include <thread>

#include <mkl.h>
#include <omp.h>



using namespace std;
size_t ThreadPool::size = -1;
static const unsigned int hash_function_nr = 5;

inline float SIMD_cosine_similarity(float *A, float *B, unsigned int size)
{
    float distance;

    SimdCosineDistance32f(A, B, size, &distance);

    return 1 - distance;
}

void random_result_preview(Join2Vec& j2v, size_t sampleSize, size_t numberOfSimilarShow, vector<string>&R, vector<string>&S){

    auto res  = j2v.getFlatResult();
    map<size_t, vector<size_t>> prev;

    for(auto el: res){
        prev[std::get<0>(el)].push_back(std::get<1>(el));
    }

    size_t i = 0;
    for(auto el: prev){
        if(i >= sampleSize) break;

        if(el.second.size()>numberOfSimilarShow) {
            i++;
            cout << R[el.first] << " - [";

            for (auto ell: el.second) {
                cout << S[ell] << ", ";
            }

            cout << "]" << endl;
        }
    }

}

string print_float(float* f, int size){
    stringstream ss;
    for(int i=0; i<size; i++){
        ss << f[i] << " ";
    }
    return ss.str();
}

void normalize(vector<myVector>&in, vector<myVector>& out){

    int dims = in[0].size();

    vector<float> max_v(dims, std::numeric_limits<float>::min()), min_v(dims, std::numeric_limits<float>::max());

    // find global, per-dimension min and max
    for(auto el: in){
        auto tmp = el.getDataVector();
        for(int i=0; i<dims; i++){
            if(tmp[i]>max_v[i]) max_v[i]=tmp[i];
            if(tmp[i]<min_v[i]) min_v[i]=tmp[i];
        }
    }

    float max_val = std::numeric_limits<float>::min(), min_val = std::numeric_limits<float>::max();

    for(int i=0; i<dims; i++){
        if(max_v[i]>max_val) max_val = max_v[i];
        if(min_v[i]<min_val) min_val = min_v[i];
    }

    float diff = max_val - min_val;

    for(auto el: in){
        auto tmp {el};
        for(int i=0; i<dims; i++){
            tmp[i] = (tmp[i] - min_val)/diff;
        }
        out.push_back(tmp);
    }

}


void get_matrix_layout(vector<myVector>& R, std::vector<myVector>& S, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&Rm, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&Sm, int VECTOR_SIZE) {
    //Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Rm;
    //Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Sm;

    Rm.resize(R.size(), VECTOR_SIZE);
    //Sm.resize(S.size(), VECTOR_SIZE); // immediately transpose
    Sm.resize(VECTOR_SIZE, S.size());

    // https://stackoverflow.com/questions/33668485/eigen-and-stdvector
    for (size_t i = 0; i < R.size(); i++) {
        Rm.row(i) = Eigen::Map<Eigen::VectorXf>(R[i].data(), VECTOR_SIZE);
        Rm.row(i).normalized();
    }

    for (size_t i = 0; i < S.size(); i++) {
        Sm.col(i) = Eigen::Map<Eigen::VectorXf>(S[i].data(), VECTOR_SIZE);
        Sm.col(i).normalized();
    }
}

int scan_benchmarks(){
    cout << "STARTED EVALUATION" << endl;

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-1000000",
            model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    auto& tp = ThreadPool::getInstance(48, false);   // second parameter true - if sequential cores, false if interleave physical and logical

    auto& topology = Topology::getInstance();

    Tests t;

    /*for(int byte_width : {1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}){
        for(int rep=0; rep<5; rep++){
            system("dropcache");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            system("dropcache");


            t.scan_columns(byte_width, 10*1000*1000);
        }
        TimeManager::getInstance().printAll(cout);
    }*/

    vector<int> offsets_2;
    offsets_2.push_back(1);
    for(int i=0; i<2048;){
        if(i<64){
            i+=2;
        } else {
            if(i<128){
                i+=4;
            } else {
                if(i<256) {
                    i+=16;
                } else {
                    if(i<512) {
                        i+=32;
                    } else {
                        if(i<1024) {
                            i+=64;
                        } else {
                            i+=128;
                        }
                    }
                }
            }
        }

        offsets_2.push_back(i);
    }

    //offsets_2.push_back(2048);
    //offsets_2.push_back(1024);
    //offsets_2.push_back(1024);

    for(auto el: offsets_2){
        cout << el << endl;
    }

    //t.scan_columns2(1000*1000*2, 5, {1,2,3,4,8,10,12,14,16,20,24,28,32,40,48,56,60,64,72,80,94,100,116,128,150,256,300,400,512,600,700,800,900,1024,1500,2048,2500,3000,3500,4096,4500,5120,6144});

    //t.scan_columns_rand(1000*1000*2, 5, {1,2,3,4,6,8,10,12,15,16,17,18,20,24,28,32,36,40,48,52,58,64,68,72,80,88,96,108,116,128,150,180,200,220,240,250,256,280,320,350,378,400,450,512,600,700,768,800,900,1024,1100,1200,1400,1536,1600,1800,2000,2048});

    //t.scan_columns_rand(1000*1000*2, 7, offsets_2);

    //t.scan_columns_rand2(1000*1000*2, 7, offsets_2);

    t.scan_columns_rand3(1000*1000*2, 7, {1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 65536});

    TimeManager::getInstance().printAll(cout);

}

int nlj_variants(){
    cout << "STARTED EVALUATION" << endl;

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-1000000",
           model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    auto& tp = ThreadPool::getInstance(48, false);   // second parameter true - if sequential cores, false if interleave physical and logical

    //auto& topology = Topology::getInstance();

    Join2Vec j2v(model_path);
    j2v.loadModel();

    auto data = j2v.load_data(data_path); 

    auto R = j2v.load_data(data_path);
    auto S = j2v.load_data(data_path);

    //typeof(data) Rm{data.begin(), data.begin()+data.size()/2};
    //typeof(data) Sm{data.begin()+data.size()/2, data.end()};

    // set if prefetch is desirable or not
    auto Rv = j2v.prefetch(R);
    auto Sv = j2v.prefetch(S);

    // PREFETCH EXPERIMENT
    /*decltype(j2v.prefetchP(R)) Rv;
    decltype(j2v.prefetchP(S)) Sv;

    for(int i=0; i<5; i++)
    {
        {
            TimeBlock t("Prefetch Time Parallel");
            Rv = j2v.prefetchP(R);
            Sv = j2v.prefetchP(S);
        }
        cout << Rv.size()+Sv.size();
        Rv.clear();
        Sv.clear();
    }*/


    /*for(int i=0; i<5; i++)
    {
        TimeBlock t("Join Time no-SIMD");
        j2v.join_cosine_only(Rv, Sv, threshold, false);
    }*/

    for(int i=0; i<5; i++)
    {
        TimeBlock t("Join Time SIMD");
        j2v.join(Rv, Sv, threshold, true);
    }

    size_t cnt_j2v_ = 0;
    for(auto res: j2v.getResult()){
        cnt_j2v_+=res.size();
    }
    cout << "J2V RES CNT = " << cnt_j2v_ << endl;


    /*{
        TimeBlock t("Join Time");
        j2v.join(Rv, Sv, threshold, false);
    }*/

    TimeManager::getInstance().printAll(cout);

    j2v.clear_result();
}

int cache_benchmarks() {
    cout << "STARTED EVALUATION COSINE CACHE" << endl;

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-1000000",
            model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    auto &tp = ThreadPool::getInstance(48,
                                       false);   // second parameter true - if sequential cores, false if interleave physical and logical

    auto &topology = Topology::getInstance();

    float threshold = 0.9;

    Tests t;

//    /**
//     * Similarity aligned, vectorized, call
//     */
//    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
//        for(size_t dim : {1,4,16,64,256, 4096, 16384}) {
//            size_t n = sqrt(num_el/dim);
//
//            t.cache_cosine_sim_vec(n, dim, true);
//            TimeManager::getInstance().printAllNano(cout);
//        }
//    }
//    cout << "--------------------TIMINGS------------------" << endl;
//    TimeManager::getInstance().printAll(cout);

//    /**
//     * Similarity aligned, non-vectorized, call
//     */
//    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
//        for(size_t dim : {4096, 16384}) {
//            size_t n = sqrt(num_el/dim);
//
//            t.cache_cosine_sim(n, dim, true);
//            TimeManager::getInstance().printAllNano(cout);
//        }
//    }
//    cout << "--------------------TIMINGS------------------" << endl;
//    TimeManager::getInstance().printAll(cout);

    /**
     * Similarity aligned, non-vectorized, unrolled for case 1
     */
    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
        for(size_t dim : {1}) {
            size_t n = sqrt(num_el/dim);

            t.cache_cosine_sim11(n, threshold, true);
            TimeManager::getInstance().printAllNano(cout);
        }
    }
    cout << "--------------------TIMINGS------------------" << endl;
    TimeManager::getInstance().printAll(cout);

//    for(size_t num_el : {256000000l, 25600000000l}){
//        for(size_t dim : {4096, 16384}) {
//            size_t n = sqrt(num_el/dim);
//
//            t.cache_cosine_sim(n, dim, true);
//            TimeManager::getInstance().printAllNano(cout);
//        }
//    }
//    cout << "--------------------TIMINGS------------------" << endl;
//    TimeManager::getInstance().printAll(cout);

//    for(size_t num_el : {256000000l, 25600000000l}){
//        for(size_t dim : {4096, 16384}) {
//            size_t n = sqrt(num_el/dim);
//
//            t.cache_cosine_sim1(n, dim, true);
//            TimeManager::getInstance().printAllNano(cout);
//        }
//    }
//    cout << "--------------------TIMINGS------------------" << endl;
//    TimeManager::getInstance().printAll(cout);

//    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
//        for(size_t dim : {1}) {
//            size_t n = sqrt(num_el/dim);
//
//            t.cache_cosine_sim_block1(n, true);
//            TimeManager::getInstance().printAllNano(cout);
//        }
//    }
//    cout << "--------------------TIMINGS------------------" << endl;
//    TimeManager::getInstance().printAll(cout);
}

int cache_benchmarks_final() {
    cout << "STARTED EVALUATION COSINE CACHE" << endl;

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-1000000",
            model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    float threshold = 0.9;

    Tests t;

    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
        for(size_t dim : {1, 4, 16, 64, 256, 4096, 16384}) {
            size_t n = sqrt(num_el/dim);

            t.cache_cosine_sim(n, dim, threshold, true);
            TimeManager::getInstance().printAllNano(cout);
        }
    }
    cout << "--------------------TIMINGS CALL------------------" << endl;
    TimeManager::getInstance().printAll(cout);
    TimeManager::getInstance().reset();

    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
        for(size_t dim : {1,4,16,64,256, 4096, 16384}) {
            size_t n = sqrt(num_el/dim);

            t.cache_cosine_sim_vec(n, dim, threshold, true);
            TimeManager::getInstance().printAllNano(cout);
        }
    }
    cout << "--------------------TIMINGS VECTORIZED------------------" << endl;
    TimeManager::getInstance().printAll(cout);
    TimeManager::getInstance().reset();

    /*for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
        for(size_t dim : {1,4,16,64,256, 4096, 16384}) {
            size_t n = sqrt(num_el/dim);

            t.cache_cosine_sim1(n, dim, threshold, true);
            TimeManager::getInstance().printAllNano(cout);
        }
    }
    cout << "--------------------TIMINGS UNROLLED------------------" << endl;
    TimeManager::getInstance().printAll(cout);
    TimeManager::getInstance().reset();*/

    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
        for(size_t dim : {1, 4, 16, 64, 256, 4096, 16384}) {
            size_t n = sqrt(num_el/dim);

            t.cache_cosine_sim_blas(n, dim, 1, threshold);
            TimeManager::getInstance().printAllNano(cout);
        }
    }
    cout << "--------------------TIMINGS GEMM------------------" << endl;
    TimeManager::getInstance().printAll(cout);
    TimeManager::getInstance().reset();

    for(auto el: t.resCnt){
        cout << el.first << ": ";

        for(auto ell : el.second){
            cout << ell << ", ";
        }

        cout << endl;
    }
}


int cache_benchmarks_final64() {
    cout << "STARTED EVALUATION COSINE CACHE" << endl;

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-1000000",
            model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    float threshold = 0.9;

    Tests t;

    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
        for(size_t dim : {1, 4, 16, 64, 256, 4096, 16384}) {
            size_t n = sqrt(num_el/dim);

            t.cache_cosine_sim64(n, dim, threshold, true);
            TimeManager::getInstance().printAllNano(cout);
        }
    }
    cout << "--------------------TIMINGS CALL------------------" << endl;
    TimeManager::getInstance().printAll(cout);
    TimeManager::getInstance().reset();

    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
        for(size_t dim : {1, 4, 16, 64, 256, 4096, 16384}) {
            size_t n = sqrt(num_el/dim);

            t.cache_cosine_sim_blas64(n, dim, 1, threshold);
            TimeManager::getInstance().printAllNano(cout);
        }
    }
    cout << "--------------------TIMINGS GEMM------------------" << endl;
    TimeManager::getInstance().printAll(cout);
    TimeManager::getInstance().reset();

    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
        for(size_t dim : {1, 4, 16, 64, 256, 4096, 16384}) {
            size_t n = sqrt(num_el/dim);

            t.cache_cosine_sim_blas_normalize_only64(n, dim, threshold, true);
            TimeManager::getInstance().printAllNano(cout);
        }
    }
    cout << "--------------------TIMINGS NORM ONLY------------------" << endl;
    TimeManager::getInstance().printAll(cout);
    TimeManager::getInstance().reset();

    for(auto el: t.resCnt){
        cout << el.first << ": ";

        for(auto ell : el.second){
            cout << ell << ", ";
        }

        cout << endl;
    }
}



int cache_benchmarks_final1() {
    cout << "STARTED EVALUATION COSINE CACHE" << endl;

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-1000000",
            model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    float threshold = 0.9;

    Tests t;

    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
        for(size_t dim : {1, 4, 16, 64, 256, 4096, 16384}) {
            size_t n = sqrt(num_el/dim);

            t.cache_cosine_sim_blas1(n, dim, 1, threshold);
            TimeManager::getInstance().printAllNano(cout);
        }
    }
    cout << "--------------------TIMINGS GEMM1------------------" << endl;
    TimeManager::getInstance().printAll(cout);
    TimeManager::getInstance().reset();

    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
        for(size_t dim : {1, 4, 16, 64, 256, 4096, 16384}) {
            size_t n = sqrt(num_el/dim);

            t.cache_cosine_sim_blas_normalize_only1(n, dim, threshold, true);
            TimeManager::getInstance().printAllNano(cout);
        }
    }
    cout << "--------------------TIMINGS NORM1 ONLY------------------" << endl;
    TimeManager::getInstance().printAll(cout);
    TimeManager::getInstance().reset();

    for(auto el: t.resCnt){
        cout << el.first << ": ";

        for(auto ell : el.second){
            cout << ell << ", ";
        }

        cout << endl;
    }
}


int cache_benchmarks_norm_only() {
    cout << "STARTED EVALUATION COSINE CACHE" << endl;

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-1000000",
            model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    float threshold = 0.9;

    Tests t;

    for(size_t num_el : {25600l,2560000l,64000000l, 256000000l, 25600000000l}){
        for(size_t dim : {1, 4, 16, 64, 256, 4096, 16384}) {
            size_t n = sqrt(num_el/dim);

            t.cache_cosine_sim_blas_normalize_only(n, dim, 1, threshold);
            TimeManager::getInstance().printAllNano(cout);
        }
    }
    cout << "--------------------TIMINGS GEMM------------------" << endl;
    TimeManager::getInstance().printAll(cout);
    TimeManager::getInstance().reset();

}



int join_benchmark() {
    cout << "STARTED EVALUATION JOIN" << endl;

    auto& tp = ThreadPool::getInstance(48, false);

    float threshold = 0.9;

    Tests t;

    t.join_BLAS(10000l, 10000l, 100, threshold);
    TimeManager::getInstance().printAll(cout);

    t.join_BLAS(100000l, 1000l, 100, threshold);
    TimeManager::getInstance().printAll(cout);

    t.join_BLAS(1000l, 100000l, 100, threshold);
    TimeManager::getInstance().printAll(cout);

    t.join_BLAS(1000000l, 1000l, 100, threshold);
    TimeManager::getInstance().printAll(cout);

    t.join_BLAS(1000l, 1000000l, 100, threshold);
    TimeManager::getInstance().printAll(cout);

    t.join_BLAS(10000l, 100000l, 100, threshold);
    TimeManager::getInstance().printAll(cout);

    t.join_BLAS(100000l, 10000l, 100, threshold);
    TimeManager::getInstance().printAll(cout);

    t.join_BLAS(100000l, 100000l, 100, threshold);
    TimeManager::getInstance().printAll(cout);

    t.join_BLAS(10000l, 1000000l, 100, threshold);
    TimeManager::getInstance().printAll(cout);

    t.join_BLAS(1000000l, 10000l, 100, threshold);
    TimeManager::getInstance().printAll(cout);

    //t.join_BLAS_batch(1000000l, 10000l, 100,threshold, make_pair(500000l, 10000l));  // 100k x 100k is the max that fits in memory (~40GB) - try also with 500kx100k (~200GB) and try also with 200kx100k (~80GB)
    //TimeManager::getInstance().printAll(cout);

    //t.join_BLAS_batch(1000000l, 100000l, 100, threshold, make_pair(200000l, 100000l));  // 100k x 100k is the max that fits in memory (~40GB) - try also with 500kx100k (~200GB)
    //TimeManager::getInstance().printAll(cout);

    //t.join_BLAS_batch(1000000l, 1000000l, 100, threshold, make_pair(500000l, 100000l)); // 100k x 100k is the max that fits in memory (~40GB) - try also with 500kx100k (~200GB)
    //TimeManager::getInstance().printAll(cout);

    cout << "--------------------TIMINGS BLAS JOIN------------------" << endl;
    TimeManager::getInstance().printAll(cout);
    cout << "--------------------TIMINGS BLAS JOIN NANO------------------" << endl;
    TimeManager::getInstance().printAllNano(cout);
    TimeManager::getInstance().reset();
}


int join_batch_sensitivity_benchmark() {
    cout << "STARTED BATCH SENSITIVITY EVALUATION JOIN" << endl;

    auto& tp = ThreadPool::getInstance(48, false);

    float threshold = 0.9;

    Tests t;

    /*for(size_t a : {5000, 10000, 50000, 100000}){
        for(size_t b : {5000, 10000, 50000, 100000}){
            t.join_BLAS_batch(100000l, 100000l, 100,threshold, make_pair(a, b));
        }
    }*/

    for(size_t a : {1000, 10000, 100000}){
        for(size_t b : {1000, 10000, 100000}){
            t.join_BLAS_batch(100000l, 100000l, 100,threshold, make_pair(a, b));
        }
    }

    cout << "--------------------TIMINGS BLAS JOIN------------------" << endl;
    TimeManager::getInstance().printAll(cout);
    cout << "--------------------TIMINGS BLAS JOIN NANO------------------" << endl;
    TimeManager::getInstance().printAllNano(cout);
    TimeManager::getInstance().reset();
}

//

int main(){
    /*cout << "STARTED EVALUATION" << endl;

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-1000000",
           model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    auto& tp = ThreadPool::getInstance(48, false);   // second parameter true - if sequential cores, false if interleave physical and logical

    auto& topology = Topology::getInstance();*/

    //scan_benchmarks();

    //cache_benchmarks();


    // TODO: a way to impose running BLAS on given threads (1) explicitly + get results/sum
    //mkl_set_num_threads(1);
    //omp_set_num_threads(1);
    //cache_benchmarks_final1();
    //cache_benchmarks_final64();
    //cache_benchmarks_final();
    //cache_benchmarks_norm_only();

    mkl_set_num_threads(48);
    omp_set_num_threads(48);
    //join_benchmark();
    join_batch_sensitivity_benchmark();
}