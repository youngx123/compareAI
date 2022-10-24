#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <memory>
#include <chrono>
#include <paddle_api.h>
#include <opencv2/opencv.hpp>

//#define ARM_PLATFORM
#ifdef ARM_PLATFORM
#include <arm_neon.h>
#else
#include "NEON_2_SSE.h"
#endif
using namespace paddle;


class liteCompareAI
{
public:
    liteCompareAI(const std::string &nb_path, std::string node, int width, int height, 
                                    int _num_thread, std::string file,  int mode);

    void  readTextFile(std::string txtfile, std::vector<std::string> &label_Lists);    


    float inference(cv::Mat img, std::string &label, float &tmp_process, float &tmp_infer, float &tmp_post);

    void neon_mean_scale(cv::Mat img, float *dout, int size, const std::vector<float> mean, const std::vector<float> scale);
    
    void pre_process(const cv::Mat &img, int width, int height, float *data);
    
    ~liteCompareAI();

    int input_size[2];             // input height and width
    int num_class = 1000;   // number of classes

private:
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
    std::string output_name ;
    std::vector<std::string> label_Lists;  

    int num_thread;
    
    int image_w;
    int image_h;

    const std::vector<float> mean = {103.53f, 116.28f, 123.675f};
    const std::vector<float> scale = {0.017429f, 0.017507f, 0.017125f};
};

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length);

inline float fast_exp(float x);
inline float sigmoid(float x);
