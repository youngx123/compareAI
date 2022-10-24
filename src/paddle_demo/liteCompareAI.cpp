#include "liteCompareAI.h"
#include <chrono>

liteCompareAI::liteCompareAI(const std::string &nb_path, std::string node, int width, int height, 
                                    int _num_thread, std::string file,  int mode)
{
        input_size[0] = height;
        input_size[1] = width;

        output_name = node;

        lite_api::MobileConfig config;
        config.set_threads(_num_thread);
        config.set_model_from_file(nb_path);

        // set opencl backend
        bool is_opencl_valid = lite_api::IsOpenCLBackendValid();
        std::cout << "is opencl backen valid :   " << (is_opencl_valid ? "true" : "false") << std::endl;
        if (is_opencl_valid)
        {
                const std::string bin_path = "./data";
                const std::string bin_name = "lite_opencl_kernel.bin";
                config.set_opencl_binary_path_name(bin_path, bin_name);

                const std::string tuned_path = "./data";
                const std::string tuned_name = "lite_opencl_tuned.bin";
                config.set_opencl_tune(lite_api::CL_TUNE_NORMAL, tuned_path, tuned_name);

                // config.set_opencl_precision(lite_api::CLPrecisionType p = lite_api::CL_PRECISION_FP32);
                config.set_opencl_precision(lite_api::CL_PRECISION_FP32);
        }

        // set poewer mode
        // const lite_api::PowerMode CPU_POWER_MODE = lite_api::PowerMode::LITE_POWER_FULL;
        // config.set_power_mode(CPU_POWER_MODE);
        std::cout << "load  predictor\n";
        predictor = lite_api::CreatePaddlePredictor<lite_api::MobileConfig>(config);

        std::cout << "end  init\n";
}


void liteCompareAI::neon_mean_scale(cv::Mat img, float *dout, int size,
                                     const std::vector<float> mean, const std::vector<float> scale)
{
        const float *din = reinterpret_cast<const float *>(img.data);
        if (mean.size() != 3 || scale.size() != 3)
        {
                std::cerr << "[ERROR] mean or scale size must equal to 3\n";
                exit(1);
        }
        float32x4_t vmean0 = vdupq_n_f32(mean[0]);
        float32x4_t vmean1 = vdupq_n_f32(mean[1]);
        float32x4_t vmean2 = vdupq_n_f32(mean[2]);
        float32x4_t vscale0 = vdupq_n_f32(scale[0]);
        float32x4_t vscale1 = vdupq_n_f32(scale[1]);
        float32x4_t vscale2 = vdupq_n_f32(scale[2]);

        float *dout_c0 = dout;
        float *dout_c1 = dout + size;
        float *dout_c2 = dout + size * 2;

        int i = 0;
        for (; i < size - 3; i += 4)
        {
                float32x4x3_t vin3 = vld3q_f32(din);
                float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
                float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
                float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
                float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
                float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
                float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
                vst1q_f32(dout_c0, vs0);
                vst1q_f32(dout_c1, vs1);
                vst1q_f32(dout_c2, vs2);

                din += 12;
                dout_c0 += 4;
                dout_c1 += 4;
                dout_c2 += 4;
        }
        for (; i < size; i++)
        {
                *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
                *(dout_c0++) = (*(din++) - mean[1]) * scale[1];
                *(dout_c0++) = (*(din++) - mean[2]) * scale[2];
        }
}

void liteCompareAI::pre_process(const cv::Mat &img, int width, int height, float *data)
{
        cv::Mat rgb_img, imgf, chw_img;
        cv::resize(img, imgf, cv::Size(width, height));
        imgf.convertTo(imgf, CV_32FC3, 1.f);      
        
        // NHWC->NCHW
        neon_mean_scale(imgf, data, width * height, mean, scale);
}

float liteCompareAI::inference(cv::Mat img, std::string &label, float &tmp_process, float &tmp_infer, float &tmp_post)
{
        std::unique_ptr<lite_api::Tensor> input_tensor = predictor->GetInput(0);
        input_tensor->Resize({1, 3,input_size[0], input_size[1]});
        auto *data = input_tensor->mutable_data<float>();

        int height = img.rows;
        int width = img.cols;

        auto process_time1 = std::chrono::steady_clock::now();
        pre_process(img, input_size[1], input_size[0], data);
        auto process_time2 = std::chrono::steady_clock::now();

        predictor->Run();

        std::unique_ptr<const lite_api::Tensor> output_tensor = predictor->GetOutput(0);
        auto *outpt = output_tensor->data<float>();

        auto infer_time = std::chrono::steady_clock::now();

        // post processing
        // softmaxt  +  get max value
        const float*  dis_after_sm = outpt + num_class;

        // activation_function_softmax(pred_res, dis_after_sm, number_class); 
        float score=-999.0;
        int pred_label;
        for (int label = 0; label < num_class; label++)
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

        std::chrono::duration<double> process_spent = process_time2 - process_time1;
        std::chrono::duration<double> infer_spent = infer_time - process_time2;
        std::chrono::duration<double> post_spent = post_time - infer_time;

        auto output_shape = output_tensor->shape();
        std::cout << "Output shape " << output_shape[0] << "  "
                  << output_shape[1] <<  std::endl;

        std::cout << "pre processing  time  :  " << process_spent.count() * 1000 << "    ms\n ";
        std::cout << "Inference time       :  " << infer_spent.count() * 1000 << "    ms\n ";
        std::cout << "post processing time :  " << post_spent.count() * 1000 << "    ms\n\n ";

        return process_spent.count() + infer_spent.count() + post_spent.count();
}

liteCompareAI::~liteCompareAI()
{
        predictor = nullptr;
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

inline float sigmoid(float x)
{
        return 1.0f / (1.0f + fast_exp(-x));
}

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length)
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
