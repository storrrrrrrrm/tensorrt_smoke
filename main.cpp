#include <fstream>

#include <opencv2/imgcodecs.hpp>

#include "smoke.hpp"
#include "trt_modulated_deform_conv.hpp"


int main(int argc, char** argv) 
{
    cv::Mat kitti_img = cv::imread("../kitti_000008.png");
    cv::Mat intrinsic = (cv::Mat_<float>(3, 3) << 
        721.5377, 0.0, 609.5593, 0.0, 721.5377, 172.854, 0.0, 0.0, 1.0);
    // SMOKE smoke("../smoke_dla34.engine", intrinsic);
    // smoke.Detect(kitti_img);
    SMOKE smoke;
    smoke.prepare(intrinsic);
    std::string onnx_path = "../smoke_dla34.onnx";
    std::string engine_path = "../smoke_dla34.engine";
    std::ifstream f(engine_path.c_str());
    bool engine_file_exist = f.good();
    if(engine_file_exist)
    {
        smoke.LoadEngine(engine_path);
        for(int i = 0;i < 100;i++)
        {
            smoke.Detect(kitti_img);
        }
        
    }
    else
    {
        smoke.LoadOnnx(onnx_path);
    }

    return 0;
}
