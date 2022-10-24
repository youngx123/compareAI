#ifndef __MNNCOMPAREAI_H__
#define __MNNCOMPAREAI_H__

#pragma once
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"
#include <opencv2/opencv.hpp>


class mnnCompareAI {
public:
        mnnCompareAI(const std::string &mnn_path, std::string node, int width, int height, 
                                        int _num_thread, std::string file,  int mode);

        void  readTextFile(std::string txtfile, std::vector<std::string> &label_Lists);    

        ~mnnCompareAI();

        float detect(cv::Mat &img, std::string &label, float &tmp_process, float &tmp_infer, float &tmp_post);

        int input_size[2] ;                    // input height and width
        int number_class = 1000;    // number of classes. 80 for COCO

        std::string input_name = "images";

private:
        std::shared_ptr<MNN::Interpreter> MNN_interpreter;
        MNN::Session *MNN_session = nullptr;
        MNN::Tensor *input_tensor = nullptr;

         std::string output_name ;
        std::vector<std::string> label_Lists;    
        int num_thread;
        int image_w;
        int image_h;

        const float mean_vals[3] = { 255.*0.485, 255.*0.456,255.* 0.406};
        const float norm_vals[3] = { 1./(255*0.229) , 1./(255*0.224 ), 1./(255*0.225) };
};

template <typename _Tp>
int activation_function_softmax( _Tp *src, _Tp *dst, int length);

inline float fast_exp(float x);
extern float sigmoid(float x);

#endif 
