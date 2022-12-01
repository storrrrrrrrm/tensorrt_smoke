#include "smoke.hpp"

#include <fstream>
#include <memory>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <NvOnnxParser.h>

using namespace std;

#define IMAGE_H 375
#define IMAGE_W 1242
#define INPUT_H 384
#define INPUT_W 1280
#define OUTPUT_H (INPUT_H / 4)
#define OUTPUT_W (INPUT_W / 4)
#define SCORE_THRESH 0.3f
#define TOPK 100

SMOKE::SMOKE()
{
    
}


SMOKE::SMOKE(const std::string& engine_path, const cv::Mat& intrinsic)
        : intrinsic_(intrinsic) 
{

}

SMOKE::~SMOKE() {
    cudaStreamDestroy(stream_);
    for (auto& buffer : buffers_) {
        cudaFree(buffer);
    }
    if (context_ != nullptr) {
        // context_->destroy();
        delete context_;
        context_ = nullptr;

        // engine_->destroy();
        delete engine_;
        engine_ = nullptr;
    }
}

//设置与模型有关的参数　分配显存内存等
void SMOKE::prepare(const cv::Mat& intrinsic)
{
    intrinsic_ = intrinsic.clone();
    buffer_size_[0] = 3 * INPUT_H * INPUT_W;
    buffer_size_[1] = TOPK * 8;
    buffer_size_[2] = TOPK;
    buffer_size_[3] = TOPK;
    cudaMalloc(&buffers_[0], buffer_size_[0] * sizeof(float));
    cudaMalloc(&buffers_[1], buffer_size_[1] * sizeof(float));
    cudaMalloc(&buffers_[2], buffer_size_[2] * sizeof(float));
    cudaMalloc(&buffers_[3], buffer_size_[3] * sizeof(float));
    image_data_.resize(buffer_size_[0]);
    bbox_preds_.resize(buffer_size_[1]);
    topk_scores_.resize(buffer_size_[2]);
    topk_indices_.resize(buffer_size_[3]);
    cudaStreamCreate(&stream_);

    // LoadEngine("../smoke_dla34.engine");
    

    // https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/models/smoke.py#L41
    base_depth_ = {28.01f, 16.32f};
    base_dims_.resize(3);  //pedestrian, cyclist, car
    base_dims_[0].x = 0.88f;
    base_dims_[0].y = 1.73f;
    base_dims_[0].z = 0.67f;
    base_dims_[1].x = 1.78f;
    base_dims_[1].y = 1.70f;
    base_dims_[1].z = 0.58f;
    base_dims_[2].x = 3.88f;
    base_dims_[2].y = 1.63f;
    base_dims_[2].z = 1.53f;    
    // Modify camera intrinsics due to scaling
    intrinsic_.at<float>(0, 0) *= static_cast<float>(INPUT_W) / IMAGE_W;
    intrinsic_.at<float>(0, 2) *= static_cast<float>(INPUT_W) / IMAGE_W;
    intrinsic_.at<float>(1, 1) *= static_cast<float>(INPUT_H) / IMAGE_H;
    intrinsic_.at<float>(1, 2) *= static_cast<float>(INPUT_H) / IMAGE_H;
}

void SMOKE::Detect(const cv::Mat& raw_img) 
{
    auto start =  std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Preprocessing
    cv::Mat img_resize;
    cv::resize(raw_img, img_resize, cv::Size(INPUT_W, INPUT_H), cv::INTER_LINEAR);
    // img_resize.convertTo(img_resize, CV_32FC3, 1.0f);
    float mean[3] {123.675f, 116.280f, 103.530f};
    float std[3] = {58.395f, 57.120f, 57.375f};
    uint8_t* data_hwc = reinterpret_cast<uint8_t*>(img_resize.data);
    float* data_chw = image_data_.data();
    for (int c = 0; c < 3; ++c) {
        for (unsigned j = 0, img_size = INPUT_H * INPUT_W; j < img_size; ++j) {
            data_chw[c * img_size + j] = (data_hwc[j * 3 + 2 - c] - mean[c]) / std[c];  //bgr2rgb
        }
    }
    
    // Do inference
    cudaMemcpyAsync(buffers_[0], image_data_.data(), buffer_size_[0] * sizeof(float), cudaMemcpyHostToDevice, stream_);
    auto compute_start =  std::chrono::high_resolution_clock::now();
    context_->executeV2(&buffers_[0]);
    // end = std::chrono::high_resolution_clock::now();
    // int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - compute_start);
    // std::cout<<"compute:"<<int_ms.count()<<",fps:"<<1000/int_ms.count()<<std::endl;
    cudaMemcpyAsync(bbox_preds_.data(), buffers_[1], buffer_size_[1] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(topk_scores_.data(), buffers_[2], buffer_size_[2] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(topk_indices_.data(), buffers_[3], buffer_size_[3] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    end = std::chrono::high_resolution_clock::now();
    int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - compute_start);
    std::cout<<"compute(ms):"<<int_ms.count()<<",fps:"<<1000/int_ms.count()<<std::endl;
    // Decoding and visualization
    PostProcess(img_resize);

    end = std::chrono::high_resolution_clock::now();
    int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout<<"one frame(ms):"<<int_ms.count()<<",fps:"<<1000/int_ms.count()<<std::endl;

}

void SMOKE::LoadEngine(const std::string& engine_path) {
    std::ifstream in_file(engine_path, std::ios::binary);
    if (!in_file.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return;
    }
    in_file.seekg(0, in_file.end);
    int length = in_file.tellg();
    in_file.seekg(0, in_file.beg);
    std::unique_ptr<char[]> trt_model_stream(new char[length]);
    in_file.read(trt_model_stream.get(), length);
    in_file.close();

    // getPluginCreator could not find plugin: MMCVModulatedDeformConv2d version: 1
    initLibNvInferPlugins(&g_logger_, "");
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_);
    assert(runtime != nullptr);
    engine_ = runtime->deserializeCudaEngine(trt_model_stream.get(), length);
    assert(engine_ != nullptr);
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);

    // runtime->destroy();
    delete runtime;
    runtime = nullptr;
}

void SMOKE::LoadOnnx(const std::string& onnx_path)
{   
    initLibNvInferPlugins(&g_logger_, "");
    
    // create runtime
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_);

    // create builder
    auto builder = unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger_));

    // create config from builder
    auto config = unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    
    // fp16精度的模型类别预测错误,会造成后处理崩溃.
    // config->setFlag(nvinfer1::BuilderFlag::kFP16);
    size_t workspace_size = (1ULL << 30);
    #if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8400
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
    #else
        config->setMaxWorkspaceSize(workspace_size);
    #endif

    // create network
    // const auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    const auto flag = 1U;
    auto network = unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));

    // create parser to fufill network
    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g_logger_));
    parser->parseFromFile(
        onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));

    // Build engine
    std::cout << "Applying optimizations and building TRT CUDA engine..." << std::endl;
    plan_ = builder->buildSerializedNetwork(*network, *config);
    if (!plan_) {
        std::cout << "Fail to create serialized network" << std::endl;
        return;
    }
    engine_ = runtime->deserializeCudaEngine(plan_->data(), plan_->size());


    //save engine file
    std::string engine_path = "../smoke_dla34.engine";
    std::cout << "Writing to " << engine_path << "..." << std::endl;
    std::ofstream file(engine_path, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<const char *>(plan_->data()), plan_->size());
}

float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void SMOKE::PostProcess(cv::Mat& input_img) 
{
    for (int i = 0; i < TOPK; ++i) 
    {
        if (topk_scores_[i] < SCORE_THRESH) {
            continue;
        }
        // https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/coders/smoke_bbox_coder.py#L52
        int class_id = static_cast<int>(topk_indices_[i] / OUTPUT_H / OUTPUT_W);
        int location = static_cast<int>(topk_indices_[i]) % (OUTPUT_H * OUTPUT_W);
        int img_x = location % OUTPUT_W;
        int img_y = location / OUTPUT_W;
        // Depth  bbox_preds_预测的是相对偏移.
        float z = base_depth_[0] + bbox_preds_[8*i] * base_depth_[1];
        // Location
        cv::Mat img_point(3, 1, CV_32FC1);
        img_point.at<float>(0) = 4.0f * (static_cast<float>(img_x) + bbox_preds_[8*i + 1]);
        img_point.at<float>(1) = 4.0f * (static_cast<float>(img_y) + bbox_preds_[8*i + 2]);
        img_point.at<float>(2) = 1.0f;
        cv::Mat cam_point = intrinsic_.inv() * img_point * z;
        float x = cam_point.at<float>(0);
        float y = cam_point.at<float>(1);
        // Dimension
        // std::cout<<"class_id:"<<class_id<<std::endl;
        // std::cout<<"w_offset:"<<bbox_preds_[8*i + 3]<<std::endl;
        float w = base_dims_[class_id].x * expf(Sigmoid(bbox_preds_[8*i + 3]) - 0.5f);
        float l = base_dims_[class_id].y * expf(Sigmoid(bbox_preds_[8*i + 4]) - 0.5f);
        float h = base_dims_[class_id].z * expf(Sigmoid(bbox_preds_[8*i + 5]) - 0.5f);
        // Orientation
        float ori_norm = sqrtf(powf(bbox_preds_[8*i + 6], 2.0f) + powf(bbox_preds_[8*i + 7], 2.0f));
        bbox_preds_[8*i + 6] /= ori_norm;  //sin(alpha)
        bbox_preds_[8*i + 7] /= ori_norm;  //cos(alpha)
        float ray = atan(x / (z + 1e-7f));
        float alpha = atan(bbox_preds_[8*i + 6] / (bbox_preds_[8*i + 7] + 1e-7f));
        if (bbox_preds_[8*i + 7] > 0.0f) {
            alpha -= M_PI / 2.0f;
        } else {
            alpha += M_PI / 2.0f;
        }
        float angle = alpha + ray;
        if (angle > M_PI) {
            angle -= 2.0f * M_PI;
        } else if (angle < -M_PI) {
            angle += 2.0f * M_PI;
        }

        // https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/bbox/structures/cam_box3d.py#L117
        //              front z
        //                   /
        //                  /
        //    (x0, y0, z1) + -----------  + (x1, y0, z1)
        //                /|            / |
        //               / |           /  |
        // (x0, y0, z0) + ----------- +   + (x1, y1, z1)
        //              |  /      .   |  /
        //              | / origin    | /
        // (x0, y1, z0) + ----------- + -------> x right
        //              |             (x1, y1, z0)
        //              |
        //              v
        //         down y
        cv::Mat cam_corners = (cv::Mat_<float>(8, 3) << 
            -w, -l, -h,     // (x0, y0, z0)
            -w, -l,  h,     // (x0, y0, z1)
            -w,  l,  h,     // (x0, y1, z1)
            -w,  l, -h,     // (x0, y1, z0)
             w, -l, -h,     // (x1, y0, z0)
             w, -l,  h,     // (x1, y0, z1)
             w,  l,  h,     // (x1, y1, z1)
             w,  l, -h);    // (x1, y1, z0)
        cam_corners = 0.5f * cam_corners;
        cv::Mat rotation_y = cv::Mat::eye(3, 3, CV_32FC1);
        rotation_y.at<float>(0, 0) = cosf(angle);
        rotation_y.at<float>(0, 2) = sinf(angle);
        rotation_y.at<float>(2, 0) = -sinf(angle);
        rotation_y.at<float>(2, 2) = cosf(angle);
        // cos, 0, sin
        //   0, 1,   0
        //-sin, 0, cos
        cam_corners = cam_corners * rotation_y.t();
        for (int i = 0; i < 8; ++i) {
            cam_corners.at<float>(i, 0) += x;
            cam_corners.at<float>(i, 1) += y;
            cam_corners.at<float>(i, 2) += z;
        }
        cam_corners = cam_corners * intrinsic_.t();
        std::vector<cv::Point2f> img_corners(8);
        for (int i = 0; i < 8; ++i) {
            img_corners[i].x = cam_corners.at<float>(i, 0) / cam_corners.at<float>(i, 2);
            img_corners[i].y = cam_corners.at<float>(i, 1) / cam_corners.at<float>(i, 2);
        }
        for (int i = 0; i < 4; ++i) {
            const auto& p1 = img_corners[i];
            const auto& p2 = img_corners[(i + 1) % 4];
            const auto& p3 = img_corners[i + 4];
            const auto& p4 = img_corners[(i + 1) % 4 + 4];
            cv::line(input_img, p1, p2, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
            cv::line(input_img, p3, p4, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
            cv::line(input_img, p1, p3, cv::Scalar(241, 101, 72), 1, cv::LINE_AA);
        }
    }

    // cv::imwrite("../result.png",input_img);
}
