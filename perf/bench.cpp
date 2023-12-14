//
// Created by sanca on 9/26/22.
//

#include <benchmark/benchmark.h>
#include <string>
#include <iostream>

#include "util/ThreadPool.h"
#include "join2vec/join2vec.h"

using namespace std;

size_t ThreadPool::size = -1;

static void BM_JoinSIMD(benchmark::State& state) {

    //state.PauseTiming();

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-100000",
            model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    auto& tp = ThreadPool::getInstance(state.range(0));

    Join2Vec j2v(model_path);
    j2v.loadModel();

    auto data = j2v.load_data(data_path);

    typeof(data) dataR = data;//{data.begin(), data.begin()+10000};
    typeof(data) dataS = data;// {data.begin(), data.begin()+10000};

    auto Rv = j2v.prefetch(dataR);
    auto Sv = j2v.prefetch(dataS);

    cout << "Data loaded!" << endl;

    //state.ResumeTiming();
    for (auto _ : state) {
        j2v.join(Rv, Sv, 0.9, true);
    }


    size_t cnt = 0;
    for(auto res: j2v.getResult()){
        cnt+=res.size();
    }
    cout << cnt << endl;
}

static void BM_JoinEqui(benchmark::State& state) {

    //state.PauseTiming();

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-100000",
            model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    auto& tp = ThreadPool::getInstance(state.range(0));

    Join2Vec j2v(model_path);
    j2v.loadModel();

    auto data = j2v.load_data(data_path);

    typeof(data) dataR = data;//{data.begin(), data.begin()+10000};
    typeof(data) dataS = data;// {data.begin(), data.begin()+10000};

    auto Rv = j2v.prefetch(dataR);
    auto Sv = j2v.prefetch(dataS);

    cout << "Data loaded!" << endl;

    //state.ResumeTiming();

    for (auto _ : state) {
        j2v.join(Rv, Sv, 0.9, true, true);
    }

    size_t cnt = 0;
    for(auto res: j2v.getResult()){
        cnt+=res.size();
    }
    cout << cnt << endl;
}

static void BM_JoinSIMD_(benchmark::State& state) {

    //state.PauseTiming();

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-100000",
            model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    auto& tp = ThreadPool::getInstance(state.range(0));

    Join2Vec j2v(model_path);
    j2v.loadModel();

    auto data = j2v.load_data(data_path);

    typeof(data) dataR = data;//{data.begin(), data.begin()+10000};
    typeof(data) dataS = data;// {data.begin(), data.begin()+10000};

    auto Rv = j2v.prefetch(dataR);
    auto Sv = j2v.prefetch(dataS);

    cout << "Data loaded!" << endl;

    //state.ResumeTiming();

    for (auto _ : state) {
        j2v.join_(Rv, Sv, 0.9, true);
    }

    size_t cnt = 0;
    for(auto res: j2v.getResult()){
        cnt+=res.size();
    }
    cout << cnt << endl;
}

static void BM_Join(benchmark::State& state) {

    //state.PauseTiming();

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-100000",
            model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    auto& tp = ThreadPool::getInstance(state.range(0));

    Join2Vec j2v(model_path);
    j2v.loadModel();

    auto data = j2v.load_data(data_path);

    typeof(data) dataR = data;// {data.begin(), data.begin()+10000};
    typeof(data) dataS = data;//{data.begin(), data.begin()+10000};

    auto Rv = j2v.prefetch(dataR);
    auto Sv = j2v.prefetch(dataS);

    cout << "Data loaded!" << endl;

    //state.ResumeTiming();

    for (auto _ : state) {
        j2v.join(Rv, Sv, 0.9, false);
    }

    size_t cnt = 0;
    for(auto res: j2v.getResult()){
        cnt+=res.size();
    }
    cout << cnt << endl;
}

static void BM_Join_(benchmark::State& state) {

    //state.PauseTiming();

    string data_path = "/scratch4/sanca/data/join2vec/test-datasets/english-words/sim-join/wiki-nodup-100000",
            model_path = "/scratch4/sanca/data/join2vec/result-embeddings/fasttext-wikipedia-big-corpus/wiki-vector.bin";

    auto& tp = ThreadPool::getInstance(state.range(0));

    Join2Vec j2v(model_path);
    j2v.loadModel();

    auto data = j2v.load_data(data_path);

    typeof(data) dataR = /*data;*/ {data.begin(), data.begin()+data.size()/2};
    typeof(data) dataS = /*data;*/{data.begin()+data.size()/2, data.end()};

    auto Rv = j2v.prefetch(dataR);
    auto Sv = j2v.prefetch(dataS);

    cout << "Data loaded!" << endl;

    //state.ResumeTiming();

    for (auto _ : state) {
        j2v.join_(Rv, Sv, 0.9, false);
    }

    size_t cnt = 0;
    for(auto res: j2v.getResult()){
        cnt+=res.size();
    }
    cout << cnt << endl;
}

// Register the function as a benchmark
BENCHMARK(BM_JoinEqui)->Unit(benchmark::kMillisecond)->Iterations(2)->Arg(48);

BENCHMARK(BM_JoinSIMD)->Unit(benchmark::kMillisecond)->Iterations(2)->Arg(48);

BENCHMARK(BM_JoinSIMD_)->Unit(benchmark::kMillisecond)->Iterations(2)->Arg(48);


//BENCHMARK(BM_Join)->Unit(benchmark::kMillisecond)->Iterations(2)->Arg(48);

//BENCHMARK(BM_Join_)->Unit(benchmark::kMillisecond)->Iterations(2)->Arg(48);


BENCHMARK_MAIN();


