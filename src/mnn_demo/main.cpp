
#include <iostream>
#include <sys/stat.h>
#include <dlfcn.h>
#include <json/json.h>
#include <fstream>
#include "mnnCompareAI.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


struct config
{
        std::string model_file;
        std::string label_file;
        std::string testDir;
        std::string saveDir;
        std::string output_node;
        int img_width;
        int img_height;
        int thread_num;
        int mode;
};

void readFileJson(std::string jsonPath, config &cfg)
{
        Json::Reader reader;
	Json::Value root;

	std::ifstream fin(jsonPath.c_str(), std::ios::binary);

	if (!fin.is_open()) {
		std::cout << "Error opening" << std::endl;
		return;
	}

	if (reader.parse(fin, root)) 
        {
		cfg.model_file = root["modelFile"].asString();
                cfg.label_file = root["label_file"].asString();
                cfg.testDir = root["testDir"].asString();
                cfg.saveDir = root["saveDir"].asString();
                cfg.output_node = root["output_node"].asString();
                cfg.img_width = root["img_width"].asInt();
                cfg.img_height = root["img_height"].asInt();
                cfg.thread_num = root["thread_num"].asInt();
                cfg.mode = root["mode"].asInt();
	}
	fin.close();
}



void Min_Max_Time(float t1, float &min_t, float &max_t)
{
        if (t1 < min_t)
        {
                min_t = t1;
        }

        if(t1 > max_t)
        {
                max_t = t1;
        }
}


void  showTime(std::string name, float mean_time, float min_time, float max_time)
{
        std::cout << name<< " --- mean time  :  " << mean_time 
                        << "\tmin  time  :  " << min_time
                        << "\tmax time  : " <<max_time <<std::endl;
}

int image_demo(mnnCompareAI &detector, const std::string imagepath, std::string saveFold)
{
        std::vector<cv::String> filenames;
        cv::glob(imagepath.c_str(), filenames, false);

        int height = detector.input_size[0];
        int width = detector.input_size[1];


        float total_preprocess = 0., total_inference = 0.,   total_post =0.;

        float min_preprocess_time=999.;
        float max_preprocess_time=0.;

        float min_inference_time = 999.;
        float  max_inference_time = 0.;

        float min_post_time = 999.; 
        float  max_post_time = 0.;


        float total_time = 0;

        int imgNum = filenames.size();
        for (auto img_name : filenames)
        {
                int index = img_name.rfind("/");
                std::string fileName = img_name.substr(index + 1, img_name.size());
                std::string save_file = saveFold + "/" + fileName;
                cv::Mat image = cv::imread(img_name);
                // image = image/255.0;
                cv::Mat img_rgb ;
                
                if (image.empty())
                {
                        fprintf(stderr, "cv::imread %s failed\n", img_name.c_str());
                        return -1;
                }

                cv::cvtColor(image, img_rgb, cv::COLOR_BGR2RGB);

                int img_widht, img_height;
                img_widht = image.rows;
                img_height = image.cols;
                int min_size = std::min(img_widht,img_height);
                float percet = 256.0/min_size;
                int resize_with, resize_height;
                resize_with = (int)(percet *img_widht );
                resize_height = (int)(percet *img_height );

                cv::Mat resize_short_img;
                cv::resize(image, resize_short_img, cv::Size(resize_with,resize_height ));
                // center crop
                int w_start = (resize_with - 224) / 2;
                int h_start = (resize_height - 224) / 2;

                cv::Rect rect_roi(w_start, h_start, 224,224);
                cv::Mat rect_img = resize_short_img(rect_roi);

                std::string pred_label;

                float tmp_preprocess, tmp_infer, tmp_post;
                float use_time = detector.detect(rect_img, pred_label,  tmp_preprocess, tmp_infer, tmp_post);

                total_preprocess +=tmp_preprocess;
                total_inference += tmp_infer;
                total_post +=tmp_post;

                Min_Max_Time(tmp_preprocess, min_preprocess_time, max_preprocess_time);
                Min_Max_Time(tmp_infer, min_inference_time, max_inference_time);
                Min_Max_Time(tmp_post , min_post_time, max_post_time);

                total_time += use_time;
                cv::putText(image, pred_label.c_str(), cv::Point(100, 100),
                                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));
                
                 cv::imwrite(save_file, image);

        }
        std::cout << imgNum << "     images,   ust time infor :    \n";
        showTime("total ", total_time / imgNum, 0, 0);
        showTime("preprocess",total_preprocess / imgNum, min_preprocess_time, max_preprocess_time);
        showTime("inference", total_inference / imgNum, min_inference_time, max_inference_time);
        showTime("post",total_post / imgNum, min_post_time, max_post_time);
        return 0;
}

int main(int argc, char **argv)
{
        std::string json_file = "./config.json";

        config cfg;
        readFileJson(json_file, cfg);
        mkdir(cfg.saveDir.c_str(), S_IRWXU);

        int mode;
        std::cout << "mnn   file :   " << cfg.model_file << " \n"
                          <<"image    height :   " << cfg.img_height << " \n"
                         << "image    width :   " << cfg.img_width << " \n"
                        << "image    width :   " << cfg.output_node << " \n"
                         << "cpu          num :  " << cfg.thread_num <<"\n"
                         << "backend   mode : " << cfg.mode << "\n";
        mnnCompareAI compare_ai = mnnCompareAI(cfg.model_file,cfg.output_node, cfg.img_width, cfg.img_height, cfg.thread_num, 
                                                                cfg.label_file, cfg.mode);
        image_demo(compare_ai, cfg.testDir, cfg.saveDir);
}
