//
// Created by sanca on 9/26/22.
//

#include "join2vec/join2vec.h"
#include <atomic>


Join2Vec::Join2Vec(const std::string &model_path) : model_filename(model_path) {}

Join2Vec::Join2Vec(j2v_parameters_t param) : model_filename(param.model_filename) {}


void Join2Vec::cluster(size_t num_clusters) {}


void Join2Vec::loadModel() {

    assert(std::experimental::filesystem::exists(model_filename) && "Model path incorrect");

    this->model.loadModel(model_filename);

    isModelLoaded = true;
}


void Join2Vec::join(vector<myVector>& R, std::vector<myVector>& S, double threshold, bool SIMD, bool equi_test) {

    assert(isModelLoaded);

    if(!equi_test) {
        if (SIMD) {
            nested_loop_join(R, S, threshold, &Join2Vec::SIMD_cosine_similarity);
        } else {
            nested_loop_join(R, S, threshold, &Join2Vec::cosine_similarity);
        }
    } else {
        if(SIMD) {
            nested_loop_join(R, S, 1.0, &Join2Vec::equality);
        } else {
            nested_loop_join(R, S, 1.0, &Join2Vec::equality);
        }
    }

}

void Join2Vec::join_cosine_only(vector<myVector>& R, std::vector<myVector>& S, double threshold, bool SIMD, bool equi_test) {

    assert(isModelLoaded);

    if(!equi_test) {
        if (SIMD) {
            nested_loop_cosine(R, S, threshold, &Join2Vec::SIMD_cosine_similarity);
        } else {
            nested_loop_join(R, S, threshold, &Join2Vec::cosine_similarity);
        }
    } else {
        if(SIMD) {
            nested_loop_join(R, S, 1.0, &Join2Vec::equality);
        } else {
            nested_loop_join(R, S, 1.0, &Join2Vec::equality);
        }
    }

}

void Join2Vec::join_(vector<myVector>& R, std::vector<myVector>& S, double threshold, bool SIMD, bool equi_test) {

    assert(isModelLoaded);

    if(!equi_test) {
        if (SIMD) {
            nested_loop_join_(R, S, threshold, &Join2Vec::SIMD_cosine_similarity);
        } else {
            nested_loop_join_(R, S, threshold, &Join2Vec::cosine_similarity);
        }
    } else {
        if (SIMD) {
            nested_loop_join_(R, S, 1.0, &Join2Vec::equality);
        } else {
            nested_loop_join_(R, S, 1.0, &Join2Vec::equality);
        }
    }

}


void Join2Vec::join(vector<string>& R, std::vector<string>& S, double threshold, bool SIMD) {

    assert(isModelLoaded);

        if(SIMD){
            nested_loop_join(R, S, threshold, &Join2Vec::SIMD_cosine_similarity);
        } else {
            nested_loop_join(R, S, threshold, &Join2Vec::cosine_similarity);
        }

}


std::vector<std::string> Join2Vec::load_data(string filename)
{
    std::vector<std::string> line_data;

    std::string line;
    std::ifstream infile(filename.c_str());

    while (std::getline(infile, line))
    {
        // std::cout << line << std::endl;
        line_data.push_back(line);
    }

    return line_data;
}


vector<myVector> Join2Vec::prefetch(vector<string>& R) {
    vector<myVector> Rv;
    Rv.reserve(R.size());

    assert(isModelLoaded);
    fasttext::Vector vec(model.getDimension());

    size_t cnt = 0;

    for(const string r : R){
        model.getWordVector(vec, r);
        Rv.push_back({vec, cnt++});
    }

    return Rv;
}


vector<myVector> Join2Vec::prefetchP(vector<string>& R) {
    auto&tp = ThreadPool::getInstance();
    std::vector< std::future<vector<myVector>> > results;

    assert(isModelLoaded);

    size_t parallelism = tp.getSize();
    size_t step = ceil(R.size()/(double)parallelism);

    size_t dims = getDimensions();

    for(size_t t = 0; t<parallelism; t++){
        size_t begin = t*step;
        size_t end;
        if(t==parallelism-1){
            end = R.size();
        } else {
            end = (t+1)*step;
        }

        results.emplace_back(
                tp.enqueue([this, begin, end, &R, dims](){
                    vector<myVector> res;
                    res.reserve(end-begin);

                    fasttext::Vector vec(dims);

                    for(size_t i = begin; i < end; i++){
                        model.getWordVector(vec, R[i]);
                        res.push_back({vec, i});
                    }

                    return res;
                })
        );
    }

    vector<myVector> ret;

    for(auto && res : results){
        const vector<myVector>& tmp = res.get();
        ret.insert(ret.end(), tmp.begin(), tmp.end());
    }

    return ret;
}


size_t Join2Vec::getDimensions() {
    return this->model.getDimension();
}


void Join2Vec::nested_loop_join(const vector<myVector>& Rv, const vector<myVector>& Sv, double threshold, float(Join2Vec::*f)(const float*, const float*,unsigned int)) {
    auto&tp = ThreadPool::getInstance();
    std::vector< std::future<vector<res_pair>> > results;

    size_t parallelism = tp.getSize();
    size_t step = ceil(Rv.size()/(double)parallelism);

    size_t dims = getDimensions();

    for(size_t t = 0; t<parallelism; t++){
        size_t begin = t*step;
        size_t end;
        if(t==parallelism-1){
            end = Rv.size();
        } else {
            end = (t+1)*step;
        }

        //cout << "RANGE FOR THREAD #" << t << ", [" << begin << "-" << end << "]" << endl;

        results.emplace_back(
                tp.enqueue([this, begin, end, threshold, &Rv, &Sv, &f, dims](){
                    vector<res_pair> res;

                    for(size_t i = begin; i < end; i++){
                        for(size_t j = 0; j < Sv.size(); j++){
                            if((this->*f)(Rv[i].data(), Sv[j].data(), dims) >= threshold){
                                res.push_back({i, j});
                            }
                        }
                    }

                    return res;
            })
        );
        //cout << "---" << endl;
    }

    for(auto && res : results){
        result.emplace_back(res.get());
    }
}

void Join2Vec::nested_loop_cosine(const vector<myVector>& Rv, const vector<myVector>& Sv, double threshold, float(Join2Vec::*f)(const float*, const float*,unsigned int)) {
    auto&tp = ThreadPool::getInstance();
    std::vector< std::future<double> > results;

    size_t parallelism = tp.getSize();
    size_t step = ceil(Rv.size()/(double)parallelism);

    size_t dims = getDimensions();

    for(size_t t = 0; t<parallelism; t++){
        size_t begin = t*step;
        size_t end;
        if(t==parallelism-1){
            end = Rv.size();
        } else {
            end = (t+1)*step;
        }

        //cout << "RANGE FOR THREAD #" << t << ", [" << begin << "-" << end << "]" << endl;

        results.emplace_back(
                tp.enqueue([this, begin, end, threshold, &Rv, &Sv, &f, dims](){
                    double acc = 0.0;

                    for(size_t i = begin; i < end; i++){
                        for(size_t j = 0; j < Sv.size(); j++){
                            acc += (this->*f)(Rv[i].data(), Sv[j].data(), dims);
                        }
                    }

                    return acc;
                })
        );
        //cout << "---" << endl;
    }

    double total = 0.0;
    for(auto && res : results){
        total += res.get();
    }
}


void Join2Vec::nested_loop_join(const vector<string>& wordsR, const vector<string>& wordsS, double threshold, float(Join2Vec::*f)(const float*, const float*,unsigned int)) {
    auto&tp = ThreadPool::getInstance();
    std::vector< std::future<vector<res_pair>> > results;

    size_t parallelism = tp.getSize();
    size_t step = ceil(wordsR.size()/(double)parallelism);

    size_t dims = getDimensions();

    for(size_t t = 0; t<parallelism; t++){
        size_t begin = t*step;
        size_t end;
        if(t==parallelism-1){
            end = wordsR.size();
        } else {
            end = (t+1)*step;
        }

        auto& model = this->model;

        results.emplace_back(
                tp.enqueue([this, begin, end, threshold, &model, &wordsR, &wordsS, &f, dims](){
                    vector<res_pair> res;

                    fasttext::Vector r(dims), s(dims);

                    for(size_t i = begin; i < end; i++){
                        model.getWordVector(r, wordsR[i]);
                        for(size_t j = 0; j < wordsS.size(); j++){
                            model.getWordVector(s, wordsS[j]);
                            if((this->*f)(r.data(), s.data(), dims) >= threshold){
                                res.push_back({i, j});
                            }
                        }
                    }

                    return res;
                })
        );
    }

    for(auto && res : results){
        result.emplace_back(res.get());
    }
}


void Join2Vec::nested_loop_join_(const vector<myVector>& Rv, const vector<myVector>& Sv, double threshold, float(Join2Vec::*f)(const float*, const float*,unsigned int)) {
    auto&tp = ThreadPool::getInstance();
    std::vector< std::future<vector<res_pair>> > results;

    size_t dims = getDimensions();

    for(size_t cnt_r = 0; cnt_r < Rv.size(); cnt_r++){
        {
            lock_guard<mutex> lock(inc);

            results.emplace_back(tp.enqueue([this, threshold, cnt_r, &Rv, &Sv, &f, dims]() {
                vector<res_pair> res;

                for (size_t cnt_s = 0; cnt_s < Sv.size(); cnt_s++) {
                    if ((this->*f)(Rv[cnt_r].data(), Sv[cnt_s].data(), dims) >=
                        threshold) {
                        res.push_back({cnt_r, cnt_s});
                    }
                }

                return res;
            }));
        }
    }

    for(auto && res : results){
        result.emplace_back(res.get());
    }
}

void Join2Vec::join_matrix(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&Rm, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &Sm, double threshold, bool SIMD){
    Eigen::MatrixXf Dm;
    Dm.resize(Rm.rows(), Sm.cols());
    {
        TimeBlock t("Dot product time");
        Dm = Rm * Sm;
        //auto RetM = (Rm*Sm).array() > threshold;
        //RetM.eval();
        //Dm.eval();
        //cout << RetM.size() << endl;
    }
    //Dm.eval();

    std::cout << "DM [" << Dm.rows() << "x" << Dm.cols() << "]" << std::endl;
    std::cout << "|DM| = " << Dm.size() << std::endl;
    std::cout << Dm.coeff(5, 55) << std::endl;
}

void Join2Vec::join_matrix(vector<myVector>& R, std::vector<myVector>& S, double threshold, bool SIMD){
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Rm;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Sm;

    int VECTOR_SIZE = getDimensions();

    cout << "# threads " << Eigen::nbThreads() << endl;

    Rm.resize(R.size(), VECTOR_SIZE);
    //Sm.resize(S.size(), VECTOR_SIZE); // immediately transpose
    Sm.resize(VECTOR_SIZE, S.size());

    // https://stackoverflow.com/questions/33668485/eigen-and-stdvector
    for(size_t i=0; i< R.size(); i++){
        Rm.row(i) = Eigen::Map<Eigen::VectorXf>(R[i].data(), VECTOR_SIZE);
        Rm.row(i).normalize();
    }

    for(size_t i=0; i< S.size(); i++){
        Sm.col(i) = Eigen::Map<Eigen::VectorXf>(S[i].data(), VECTOR_SIZE);
        Sm.col(i).normalize();
    }

    std::cout << "The matrix Rm is of size "
              << Rm.rows() << "x" << Rm.cols() << std::endl;
    std::cout << "It has " << Rm.size() << " coefficients" << std::endl;

    std::cout << "The matrix Sm is of size "
              << Sm.rows() << "x" << Sm.cols() << std::endl;
    std::cout << "It has " << Sm.size() << " coefficients" << std::endl;


    //std::cout << Rm.coeff(100, 95) << std::endl;
    //std::cout << Sm.coeff(5, 182) << std::endl;

    //Sm.transposeInPlace();    // no need, already loaded so
    assert((Rm.cols() == Sm.rows()) or (Rm.cols() == Sm.cols()));  // dimension check, whether there is a multiplication potential
    if(Rm.cols() != Sm.rows()){
        if(Rm.cols() == Sm.cols()){
            Sm.transposeInPlace();
            cout << "[INFO] TRANSPOSE IN PLACE HAPPENED" << endl;
        }
        assert(Rm.cols() == Sm.rows());
    }

    Eigen::MatrixXf Dm;
    Dm.resize(R.size(), S.size());

    {
        TimeBlock t("Dot product time");
        Dm = Rm * Sm;
        //auto RetM = (Rm*Sm).array() > threshold;
        //RetM.eval();
        //Dm.eval();
        //cout << RetM.size() << endl;
    }
    //Dm.eval();

    auto&tp = ThreadPool::getInstance();
    std::vector< std::future<vector<res_pair>> > results;

    size_t parallelism = tp.getSize();
    size_t step = ceil(Dm.rows()/(double)parallelism);
    size_t cols = Dm.cols();

    {
        TimeBlock t("Threshold Eval time");

        for (size_t t = 0; t < parallelism; t++) {
            size_t begin = t * step;
            size_t end;
            if (t == parallelism - 1) {
                end = Dm.rows();
            } else {
                end = (t + 1) * step;
            }

            results.emplace_back(
                    tp.enqueue([this, begin, end, threshold, cols, &Dm]() {
                        vector<res_pair> res;

                        for (size_t i = begin; i < end; i++) {
                            for (size_t j = 0; j < cols; j++) {
                                if (Dm.coeff(i, j) >= threshold) {
                                    res.push_back({i, j});
                                }
                            }
                        }

                        return res;
                    })
            );
        }
    }

    size_t cnt_t = 0;

    for(auto && res : results){
        auto tmp = res.get();
        cnt_t += tmp.size();
        result.emplace_back(tmp);
    }

    std::cout << "DM [" << Dm.rows() << "x" << Dm.cols() << "]" << std::endl;
    std::cout << "|DM| = " << Dm.size() << std::endl;
    std::cout << cnt_t << endl;
}

void Join2Vec::join_matrix(vector<vector<double>>& R, std::vector<vector<double>>& S, double threshold, bool SIMD){
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rm;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Sm;

    int VECTOR_SIZE = getDimensions();

    cout << "# threads " << Eigen::nbThreads() << endl;

    Rm.resize(R.size(), VECTOR_SIZE);
    //Sm.resize(S.size(), VECTOR_SIZE); // immediately transpose
    Sm.resize(VECTOR_SIZE, S.size());

    // https://stackoverflow.com/questions/33668485/eigen-and-stdvector
    for(size_t i=0; i< R.size(); i++){
        Rm.row(i) = Eigen::Map<Eigen::VectorXd>(R[i].data(), VECTOR_SIZE);
        Rm.row(i).normalize();
    }

    for(size_t i=0; i< S.size(); i++){
        Sm.col(i) = Eigen::Map<Eigen::VectorXd>(S[i].data(), VECTOR_SIZE);
        Sm.col(i).normalize();
    }

//    std::cout << "The matrix Rm is of size "
//              << Rm.rows() << "x" << Rm.cols() << std::endl;
//    std::cout << "It has " << Rm.size() << " coefficients" << std::endl;
//
//    std::cout << "The matrix Sm is of size "
//              << Sm.rows() << "x" << Sm.cols() << std::endl;
//    std::cout << "It has " << Sm.size() << " coefficients" << std::endl;


    //std::cout << Rm.coeff(100, 95) << std::endl;
    //std::cout << Sm.coeff(5, 182) << std::endl;

    //Sm.transposeInPlace();    // no need, already loaded so
    assert((Rm.cols() == Sm.rows()) or (Rm.cols() == Sm.cols()));  // dimension check, whether there is a multiplication potential
    if(Rm.cols() != Sm.rows()){
        if(Rm.cols() == Sm.cols()){
            Sm.transposeInPlace();
            cout << "[INFO] TRANSPOSE IN PLACE HAPPENED" << endl;
        }
        assert(Rm.cols() == Sm.rows());
    }

    Eigen::MatrixXd Dm;
    Dm.resize(R.size(), S.size());

    {
        TimeBlock t("Dot product time");
        Dm = Rm * Sm;
        //auto RetM = (Rm*Sm).array() > threshold;
        //RetM.eval();
        Dm.eval();
        //cout << RetM.size() << endl;
    }
    //Dm.eval();

    std::cout << "DM [" << Dm.rows() << "x" << Dm.cols() << "]" << std::endl;
    std::cout << "|DM| = " << Dm.size() << std::endl;
    std::cout << Dm.coeff(5, 55) << std::endl;
}

//https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
void Join2Vec::join_matrixP(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&Rm, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &Sm, double threshold, bool SIMD){
    auto&tp = ThreadPool::getInstance();
    std::vector< std::future<vector<res_pair>> > results;

    assert(isModelLoaded);

    size_t parallelism = tp.getSize();

    /*
     * BLOCK-LEVEL PARALLELISM -> partition the rows/columns to obtain block-matrix formulation
     * It could be fixed-size too (and then adapt the loop)
     */
    size_t step_R = ceil(Rm.rows()/(double)parallelism);
    size_t step_S = ceil(Sm.cols()/(double)parallelism);

    for(size_t t = 0; t<parallelism; t++){
        size_t begin_R = t*step_R;
        size_t begin_S = t*step_S;

        size_t end_R, end_S;
        if(t==parallelism-1){
            end_R = Rm.rows();
            end_S = Sm.cols();
        } else {
            end_R = (t+1)*step_R;
            end_S = (t+1)*step_S;
        }

        results.emplace_back(
                tp.enqueue([this, begin_R, end_R, &Rm, &Sm, threshold](){

                    //Eigen::MatrixXf Dmp;
                    //Dmp.resize(end_R-begin_R, Sm.cols());
                    auto& Dmp = Rm.block(begin_R,0,end_R-begin_R,Rm.cols())*Sm;

                    vector<res_pair> res;

                    for(size_t i = 0; i < Dmp.rows(); i++){
                        for(size_t j = 0; j < Dmp.cols(); j++){
                            if(Dmp.coeff(i, j) >= threshold){
                                res.push_back({begin_R+i, j});
                            }
                        }
                    }
                    cout << "DONE!" << endl;
                    return res;
                })
        );
    }

    for(auto && res : results){
        result.emplace_back(res.get());
    }
}


