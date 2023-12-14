//
// Created by sanca on 9/21/23.
//

#ifndef J2V_SCAN_H
#define J2V_SCAN_H

#include <iostream>
#include <vector>
#include <array>
#include "myVector.h"

class Scan{
private:

public:
    void generateColumn32(size_t elements, std::vector<float>& col);
    void generateVecColumn(size_t elements, size_t vec_size, std::vector<std::vector<float>>& vec);

    size_t count1vec(std::vector<std::vector<float>>& vec);
    size_t count1col(std::vector<float>& col);

    double sum();

    // experiment with different vector widths (break points at L1, L2, L3 expected)
};

#endif //J2V_SCAN_H
