//
// Created by sanca on 10/21/22.
//

#ifndef J2V_MYVECTOR_H
#define J2V_MYVECTOR_H

#include <vector>
#include "fasttext/fasttext.h"

using namespace std;

class myVector : public fasttext::Vector{
public:
    size_t id;

    myVector(fasttext::Vector& v) : fasttext::Vector(v) {}
    myVector(const fasttext::Vector& v) : fasttext::Vector(v) {}

    myVector(fasttext::Vector& v, size_t id_) : fasttext::Vector(v), id(id_) {}
    myVector(const fasttext::Vector& v, size_t id_) : fasttext::Vector(v), id(id_) {}

    decltype(data_) getDataVector(){
        return data_;
    }

    void setId(size_t id_) {
        id = id_;
    }
};

#endif //J2V_MYVECTOR_H
