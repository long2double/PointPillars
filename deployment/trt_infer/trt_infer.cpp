#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <chrono>

#include "include/common_trt.h"
#include "include/io_trt.hpp"
#include "include/voxelization_trt.h"
#include "include/utils.h"
#include "include/nms_trt.h"

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger           
{
    public:
        Severity reportableSeverity;

        Logger(Severity severity = Severity::kINFO):
            reportableSeverity(severity) {}
        void log(Severity severity, const char* msg) noexcept override
        {
            switch(severity) {
                case Severity::kINTERNAL_ERROR:
                case Severity::kERROR:
                case Severity::kWARNING:
                    std::cout << msg << std::endl;
                    break;
                case Severity::kINFO:
                case Severity::kVERBOSE:
                    // Optionally ignore or handle less severe messages
                    break;
            }
        }
} gLogger;

// 简化错误处理
#define CHECK(status) \
    if (status != 0) \
    { \
        std::cerr << "Cuda failure: " << status << std::endl; \
        abort(); \
    }

std::pair<nvinfer1::ICudaEngine*, nvinfer1::IExecutionContext*> initializeTensorRTComponents(const std::string& engineFilePath) {
    // 支持插件(scatterND)
    // https://github.com/onnx/onnx-tensorrt/issues/597
    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");

    // 读取序列化的引擎
    auto engineData = readEngineFile(engineFilePath);

    // 创建运行时和引擎
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());

    // 创建执行上下文
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    return {engine, context};
}

void trtInfer(float* d_voxels, int* d_coors, int* d_num_points_per_voxel, const int pillar_num,
              int& max_points, nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context, 
              std::vector<float>& output){

    // 获取输入和输出的绑定索引
    // int inputPillarsIndex = engine->getBindingIndex("input_pillars");
    // int inputCoorsBatchIndex = engine->getBindingIndex("input_coors_batch");
    // int inputNpointsPerPillarIndex = engine->getBindingIndex("input_npoints_per_pillar");
    // int outputIndex = engine->getBindingIndex("output_x");

    // 使用CUDA分配设备内存
    void* inputPillarsDevice;
    void* inputCoorsBatchDevice;
    void* inputNpointsPerPillarDevice;
    CHECK(cudaMalloc(&inputPillarsDevice, pillar_num * max_points * sizeof(Point)));
    CHECK(cudaMalloc(&inputCoorsBatchDevice, pillar_num * 4 * sizeof(int)));
    CHECK(cudaMalloc(&inputNpointsPerPillarDevice, pillar_num * sizeof(int)));

    // 将数据从设备复制到设备
    CHECK(cudaMemcpy(inputPillarsDevice, d_voxels, pillar_num * max_points * sizeof(Point), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(inputCoorsBatchDevice, d_coors, pillar_num * 4 * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(inputNpointsPerPillarDevice, d_num_points_per_voxel, pillar_num * sizeof(int), cudaMemcpyDeviceToDevice));
    cudaFree(d_voxels);
    cudaFree(d_coors);
    cudaFree(d_num_points_per_voxel);

    // 分配输出设备内存
    void* outputDevice;
    CHECK(cudaMalloc(&outputDevice, output.size() * sizeof(float))); 

    // 设置输入张量的维度
    nvinfer1::Dims inputPilllarDims, inputCoorsDims, inputNpointsPerPillarDims; // 您期望的输入维度
    inputPilllarDims.nbDims = 3; // 维度数
    inputPilllarDims.d[0] = pillar_num; // 每个维度的大小
    inputPilllarDims.d[1] = max_points;
    inputPilllarDims.d[2] = sizeof(Point) / sizeof(float);

    inputCoorsDims.nbDims = 2; // 维度数
    inputCoorsDims.d[0] = pillar_num; // 每个维度的大小
    inputCoorsDims.d[1] = 4;

    inputNpointsPerPillarDims.nbDims = 1; // 维度数
    inputNpointsPerPillarDims.d[0] = pillar_num; // 每个维度的大小

    // 在推理之前设置输入张量的维度
    // if (!context->setBindingDimensions(inputPillarsIndex, inputPilllarDims)) {
    //     // 处理错误，设置维度失败
    //     std::cout << "setBindingDimensions error \n";
    // }
    // if (!context->setBindingDimensions(inputCoorsBatchIndex, inputCoorsDims)) {
    //     std::cout << "setBindingDimensions error \n";
    // }
    // if (!context->setBindingDimensions(inputNpointsPerPillarIndex, inputNpointsPerPillarDims)) {
    //     std::cout << "setBindingDimensions error \n";
    // }
    context->setInputShape("input_pillars", inputPilllarDims);
    context->setInputShape("input_coors_batch", inputCoorsDims);
    context->setInputShape("input_npoints_per_pillar", inputNpointsPerPillarDims);

    // 创建输入和输出数据缓冲区指针数组
    context->setTensorAddress("input_pillars", inputPillarsDevice);
    context->setTensorAddress("input_coors_batch", inputCoorsBatchDevice);
    context->setTensorAddress("input_npoints_per_pillar", inputNpointsPerPillarDevice);
    context->setTensorAddress("output_x", outputDevice);
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // 执行推理
    context->enqueueV3(stream);

    CHECK(cudaStreamSynchronize(stream));
    // Release stream
    CHECK(cudaStreamDestroy(stream));

    // 如果需要，将输出数据从设备复制回主机
    cudaMemcpy(output.data(), outputDevice, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(inputPillarsDevice);
    cudaFree(inputCoorsBatchDevice);
    cudaFree(inputNpointsPerPillarDevice);
    cudaFree(outputDevice);
}

void postProcessing(std::vector<float>& output, int& num_class, float& nms_thr, float& score_thr, 
                     int& max_num, std::vector<Box3dfull>& bboxes_full){
    std::vector<Box2d> bboxes_2d;
    std::vector<Box3d> bboxes_3d;
    std::vector<std::vector<float>> scores_list;
    std::vector<float> direction_list;
    decodeDetResults(output, num_class, bboxes_2d, bboxes_3d, scores_list, direction_list);

    std::vector<Box3dfull> bboxes_3d_nms;
    for (int i = 0; i < num_class; i++){
        std::vector<int> score_filter_inds;
        std::vector<float> scores;
        filterByScores(i, scores_list, score_thr, score_filter_inds, scores);
        std::vector<Box2d> bboxes_2d_filtered;
        std::vector<Box3d> bboxes_3d_filtered;
        std::vector<float> direction_filtered;
        obtainBoxByInds(score_filter_inds, bboxes_2d, bboxes_2d_filtered, bboxes_3d, bboxes_3d_filtered, 
                        direction_list, direction_filtered);
        
        std::vector<int> nms_filter_inds;
        nms(bboxes_2d_filtered, scores, nms_thr, nms_filter_inds);

        for (const auto ind : nms_filter_inds){
            Box3dfull box3d_full;
            box3d_full.x = bboxes_3d_filtered[ind].x;
            box3d_full.y = bboxes_3d_filtered[ind].y;
            box3d_full.z = bboxes_3d_filtered[ind].z;
            box3d_full.w = bboxes_3d_filtered[ind].w;
            box3d_full.l = bboxes_3d_filtered[ind].l;
            box3d_full.h = bboxes_3d_filtered[ind].h;
            float limited_theta = limitPeriod(bboxes_3d_filtered[ind].theta);
            box3d_full.theta = (1.f - direction_filtered[ind]) * M_PI + limited_theta;
            box3d_full.score = scores[ind];
            box3d_full.label = i;
            bboxes_3d_nms.push_back(box3d_full);
        }
    }
    getTopkBoxes(bboxes_3d_nms, max_num, bboxes_full);
}

int main(int argc, char *argv[]) {
    // if (argc != 3){
    //     std::cerr << "Usage: " << argv[0] << " your_point_cloud_path your_trt_path\n";
    //     return 1;
    // }

    // 0. read data
    std::vector<Point> points_ori, points; 
    std::string file_path = "../../../dataset/demo_data/val/000134.bin";
    bool read_data_ok = readPoints(file_path, points_ori);
    if (!read_data_ok) return 0;
    pointCloudFiler(points_ori, points);

    std::vector<float> voxel_size = {0.16, 0.16, 4};
    std::vector<float> coors_range = {0, -39.68, -3, 69.12, 39.68, 1};
    int max_points = 32;
    int max_voxels = 40000;
    int NDim = 3;

    int* d_num_points_per_voxel = nullptr;
    cudaMalloc((void**)&d_num_points_per_voxel, max_voxels * sizeof(int));
    cudaMemset(d_num_points_per_voxel, 0, max_voxels * sizeof(int));
    float* d_voxels = nullptr;
    cudaMalloc((void**)&d_voxels, max_voxels * max_points * sizeof(Point));
    cudaMemset(d_voxels, 0.f, max_voxels * max_points * sizeof(Point));
    int* d_coors = nullptr;
    cudaMalloc((void**)&d_coors, max_voxels * NDim * sizeof(int));
    cudaMemset(d_coors, 0, max_voxels * NDim * sizeof(int));

    // 1. voxelization
    int voxel_num = voxelizeGpu(points, voxel_size, coors_range, max_points, max_voxels, d_voxels, d_coors, d_num_points_per_voxel, NDim);
    int* d_coors_padded = nullptr;
    cudaMalloc((void**)&d_coors_padded, voxel_num * (NDim + 1) * sizeof(int));
    cudaMemset(d_coors_padded, 0, voxel_num * (NDim + 1) * sizeof(int));
    padCoorsGPU(d_coors, d_coors_padded, voxel_num);
    cudaFree(d_coors);

    // 2. trt inference
    const std::string trt_path = "/workspace/PointPillars/pretrained/model.engine";
    int num_class = 3, num_box = 100;
    std::vector<float> output(num_box * (7 + num_class + 1));
    auto components = initializeTensorRTComponents(trt_path);
    nvinfer1::ICudaEngine* engine = components.first;
    nvinfer1::IExecutionContext* context = components.second;
    trtInfer(d_voxels, d_coors_padded, d_num_points_per_voxel, voxel_num, max_points, engine, context, output);
    
    // 3. post processing
    float nms_thr = 0.01, score_thr = 0.1;
    int max_num = 50;
    std::vector<Box3dfull> bboxes;
    postProcessing(output, num_class, nms_thr, score_thr, max_num, bboxes);

    // 4. write results to file
    writeFile(bboxes, "../trt.txt");

    // context->destroy();
    // engine->destroy();
    return 0;
}