//
// Created by sanca on 10/22/22.
//

#include "TimeBlock.h"

TimeBlock::~TimeBlock() {
    auto diff = chrono::high_resolution_clock::now() - start;

    TimeManager::getInstance().entries[name].push_back(diff);

    if(verbose) std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << std::endl;
}