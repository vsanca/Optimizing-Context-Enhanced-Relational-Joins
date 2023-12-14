//
// Created by sanca on 10/17/22.
//

#ifndef J2V_TIMEBLOCK_H
#define J2V_TIMEBLOCK_H

#include <stdlib.h>
#include <chrono>
#include <string>
#include <map>
#include <iostream>
#include <vector>
#include <sstream>

using namespace std;

class TimeManager;

class TimeBlock
{
private:
    decltype(chrono::high_resolution_clock::now()) start;
    decltype(chrono::high_resolution_clock::now()) end;
    string name;
    bool verbose;

    friend class TimeManager;
public:
    TimeBlock(string name, bool verbose=false){
        this->name = name;
        start = chrono::high_resolution_clock::now();
    }
    ~TimeBlock();
};

class TimeManager
{
public:
    static TimeManager& getInstance(){
        static TimeManager instance;
        return instance;
    }

    map<string, vector<decltype(chrono::high_resolution_clock::now()-chrono::high_resolution_clock::now())>>& getEntries(){
        return entries;
    }

    vector<decltype(chrono::high_resolution_clock::now()-chrono::high_resolution_clock::now())>& getEntry(string s){
        return entries[s];
    }

    void printAll(decltype(cout)& ostream){
        stringstream ss;

        for(const auto el : entries){
            ss << "[ " << el.first << " ] = ";

            for(auto els : el.second)
                ss << std::chrono::duration_cast<chrono::milliseconds>(els).count() << "ms ";

            ss << endl;
        }

        ostream << ss.str();
    }

    void printAllNano(decltype(cout)& ostream){
        stringstream ss;

        for(const auto el : entries){
            ss << "[ " << el.first << " ] = ";

            for(auto els : el.second)
                ss << std::chrono::duration_cast<chrono::nanoseconds>(els).count() << "ns ";

            ss << endl;
        }

        ostream << ss.str();
    }

    void reset(){
        for(auto el : entries){
            el.second.clear();
        }

        entries.clear();
    }

private:
    TimeManager(){}
    map<string, vector<decltype(chrono::high_resolution_clock::now()-chrono::high_resolution_clock::now())>> entries;
    friend class TimeBlock;
};



#endif //J2V_TIMEBLOCK_H
