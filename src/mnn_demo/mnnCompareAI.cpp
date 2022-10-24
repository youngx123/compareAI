#include "mnnCompareAI.hpp"
#include <math.h>
#include <unistd.h>
#include <fstream>
using namespace std;


mnnCompareAI::mnnCompareAI(const std::string &mnn_path, std::string node, int width, int height, 
                                int _num_thread, std::string file,  int mode)
{
        input_size[0] = height;
        input_size[1] = width;

        output_name = node;

        std::string txtfile = file;
        readTextFile(txtfile, label_Lists);

        num_thread = _num_thread;

        MNN_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
        int forward;
        if (mode == 0)
        {
                std::cout << "forward CPU  as backend\n";
                forward = MNN_FORWARD_CPU;
        }
        else if (mode == 1)
        {
                std::cout << "forward OPENCL  as backend\n";
                forward = MNN_FORWARD_OPENCL;
        }
        else if (mode == 2)
        {
                std::cout << "forward VULKAN  as backend\n";
                forward = MNN_FORWARD_VULKAN;
        }
        sleep(5);
        MNN::ScheduleConfig config;
        config.numThread = num_thread;
        config.type = static_cast<MNNForwardType>(forward);
        MNN::BackendConfig backendConfig;
        backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
        config.backendConfig = &backendConfig;

        MNN_session = MNN_interpreter->createSession(config);

        input_tensor = MNN_interpreter->getSessionInput(MNN_session, nullptr);
}


/*
read label file
*/
void  mnnCompareAI::readTextFile(std::string txtfile, std::vector<std::string> &labelLists)
{
        ifstream ifs;
        ifs.open(txtfile.c_str());   //若打印为乱码，将txt文件打开另存为，编码方式选为ANSI
        string str;

        while (getline(ifs, str))
        {
                // std::cout<<str<<std::endl;
                labelLists.push_back(str);
        }
        ifs.close();

        std::cout<< "label list :"<< labelLists.size()<<"\n";
}

float mnnCompareAI::detect(cv::Mat &raw_image, std::string &label, float &tmp_process, float &tmp_infer, float &tmp_post)
{
        if (raw_image.empty())
        {
                std::cout << "image is empty ,please check!" << std::endl;
                return -1;
        }

        image_h = raw_image.rows;
        image_w = raw_image.cols;
        cv::Mat image;
        cv::resize(raw_image, image, cv::Size(input_size[1], input_size[0]));

        // inferene time
        auto process_time1  = chrono::steady_clock::now();
        MNN_interpreter->resizeTensor(input_tensor, {1, 3, input_size[0], input_size[1]});
        MNN_interpreter->resizeSession(MNN_session);

        std::shared_ptr<MNN::CV::ImageProcess> pretreat(
                MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::RGB, mean_vals, 3, norm_vals, 3));
        pretreat->convert(image.data, input_size[1], input_size[0], image.step[0], input_tensor);

        auto inference_start = chrono::steady_clock::now();
        // run network
        MNN_interpreter->runSession(MNN_session);

        // get output data
        MNN::Tensor *tensor_preds = MNN_interpreter->getSessionOutput(MNN_session, output_name.c_str());

        // copy to host spent about 0.3 ms
        MNN::Tensor tensor_preds_host(tensor_preds, tensor_preds->getDimensionType());
        tensor_preds->copyToHostTensor(&tensor_preds_host);

        auto inference_end = chrono::steady_clock::now();
        // post processing
        // softmaxt  +  get max value
        float *dis_after_sm;// = new float[number_class];
        MNN::Tensor *bbox_pred = &tensor_preds_host;
        float*  pred_res = bbox_pred->host<float>() + number_class;
        // activation_function_softmax(pred_res, dis_after_sm, number_class); 
        dis_after_sm = bbox_pred->host<float>() + number_class;

        float score=-999.0;
        int pred_label;
        for (int label = 0; label < number_class; label++)
        {
                if (dis_after_sm[label] > score)
                {
                        score = dis_after_sm[label];
                        std::cout<<score << std::endl;
                        pred_label = label;
                }
        }
        label = label_Lists.at(pred_label);

        auto post_time = std::chrono::steady_clock::now();
        
        std::chrono::duration<double> process_spent = inference_start - process_time1;
        std::chrono::duration<double> infer_spent = inference_end - inference_start;
        std::chrono::duration<double> post_spent = post_time - inference_end;

        std::cout << "input shape  :  " << input_tensor->shape()[0] << "   "
                << input_tensor->shape()[1] << "    "
                << input_tensor->shape()[2] << "    "
                << input_tensor->shape()[3] << " \n";

        std::cout << "output shape  :  " << tensor_preds->shape()[0] << "   "
                << tensor_preds->shape()[1] << " \n\n";

        std::cout << "pre processing  time  :  " << process_spent.count() * 1000 << "    ms\n ";
        std::cout << "Inference   time       :  " << infer_spent.count() * 1000 << "    ms\n ";
        std::cout << "post processing time :  " << post_spent.count() * 1000 << "    ms\n\n ";
        tmp_process = process_spent.count() * 1000 ; 
        tmp_infer = infer_spent.count() * 1000 ;
        tmp_post = post_spent.count() * 1000 ;
        return tmp_process + tmp_infer + tmp_post;
}

inline float fast_exp(float x)
{
        union
        {
                uint32_t i;
                float f;
        } v{};
        v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
        return v.f;
}

mnnCompareAI::~mnnCompareAI()
{
        MNN_interpreter->releaseModel();
        MNN_interpreter->releaseSession(MNN_session);
}

inline float sigmoid(float x)
{
        return 1.0f / (1.0f + fast_exp(-x));
        // return 1.0f / (1.0f + exp(-x));
}

template <typename _Tp>
int activation_function_softmax(_Tp *src, _Tp *dst, int length)
{
        const _Tp alpha = *std::max_element(src, src + length);
        _Tp denominator{0};

        for (int i = 0; i < length; ++i)
        {
                dst[i] = fast_exp(src[i] - alpha);
                denominator += dst[i];
        }

        for (int i = 0; i < length; ++i)
        {
                dst[i] /= denominator;
        }

        return 0;
}
