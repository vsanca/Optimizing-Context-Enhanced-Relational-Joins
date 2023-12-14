//
// Created by sanca on 9/27/22.
// FROM: https://github.com/progschj/ThreadPool/blob/master/ThreadPool.h
//

#ifndef J2V_THREADPOOL_H
#define J2V_THREADPOOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <map>
#include <sstream>

using namespace std;

class Topology{
private:
    int HWConcurrency;

    Topology(){
        HWConcurrency = std::thread::hardware_concurrency();
    }

    // https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po
    string exec(string cmd){

        std::array<char, 128> buffer;
        std::string result;

        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);

        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        return result;
    }

    int getPhysCore(int core, string delim=","){
        assert(core < HWConcurrency);

        string cmd = "cat /sys/devices/system/cpu/cpu"+to_string(core)+"/topology/thread_siblings_list";

        string res = exec(cmd);

        string ret = res.substr(0, res.find(delim));

        return std::stoi(ret);
    }

public:
    static Topology& getInstance(){
        static Topology instance;
        return instance;
    }

    vector<int> getDefaultCPU(){
        vector<int> ret;
        for(int i=0; i<HWConcurrency; i++){
            ret.push_back(i);
        }

        cout << "TOPOLOGY WITH " << HWConcurrency << " CORES" << endl;

        stringstream ss;

        for(auto el: ret){
            ss << el << " ";
        }
        cout << "HW CPU CORES ORDERED SEQUENTIALLY [POTENTIALLY PHYISCAL FIRST, THEN VIRTUAL] " << ss.str() << endl;

        return ret;
    }

    vector<int> getPhysVirtInterleavedCPU(){
        map<int, vector<int>> mapping;
        vector<int> ret;

        for(int i=0; i<HWConcurrency; i++){
            mapping[getPhysCore(i)].push_back(i);
        }

        for(auto el: mapping){
            for(auto ell : el.second){
                ret.push_back(ell);
            }
        }

        cout << "TOPOLOGY WITH " << HWConcurrency << " CORES" << endl;

        stringstream ss;

        for(auto el: ret){
            ss << el << " ";
        }
        cout << "HW CPU CORES ORDERED [PHYSICAL-VIRTUAL] " << ss.str() << endl;

        return ret;
    }
};

class ThreadPool {
public:
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();
    static ThreadPool& getInstance(){
        if(size==-1)
            size = std::thread::hardware_concurrency();
        return getInstance(size);
    }
    static ThreadPool& getInstance(size_t, bool=false);
    static size_t size;
    size_t getSize() {
        return workers.size();
    }
private:
    ThreadPool(size_t, bool);
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    // the task queue
    std::queue< std::function<void()> > tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

inline ThreadPool& ThreadPool::getInstance(size_t threads, bool sequential_order) {
    size = threads;
    static ThreadPool instance(threads, sequential_order);
    assert(instance.workers.size()==threads);
    return instance;
}

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads, bool seq_order)
        :   stop(false)
{
    vector<int> corePinning;
    if(seq_order){
        corePinning = Topology::getInstance().getDefaultCPU();
    } else {
        corePinning = Topology::getInstance().getPhysVirtInterleavedCPU();
    }

    for(size_t i = 0;i<threads;++i) {
        workers.emplace_back(
                [this] {
                    for (;;) {
                        std::function<void()> task;

                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock,
                                                 [this] { return this->stop || !this->tasks.empty(); });
                            if (this->stop && this->tasks.empty())
                                return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }

                        task();
                    }
                }
        );

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        CPU_SET(corePinning[i]%std::thread::hardware_concurrency(), &cpuset);
        int rc = pthread_setaffinity_np(workers[i].native_handle(),
                                        sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
        }
    }

}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
-> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

#endif //J2V_THREADPOOL_H
