//
// Created by sanca on 9/21/23.
//

#include "scan.h"

void Scan::generateColumn32(size_t elements, std::vector<float> &col) {
    col.resize(elements);

    for(size_t i=0; i<elements; i++){
        col.push_back(float(i));
    }
}



void Scan::generateVecColumn(size_t elements, size_t vec_size, std::vector<std::vector<float>> &vec) {
    vec.resize(elements);

    for(size_t i=0; i<elements; i++){
        std::vector<float> t;
        t.resize(vec_size);
        for(size_t j=0; j<vec_size; j++){
            t.push_back(j);
        }
        vec.push_back(t);
    }
}