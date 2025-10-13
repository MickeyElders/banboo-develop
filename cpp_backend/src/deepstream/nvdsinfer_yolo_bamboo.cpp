#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>

// YOLO解析函数 - 必须与配置文件中parse-bbox-func-name一致
extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // 查找output层
    const NvDsInferLayerInfo* outputLayer = nullptr;
    for (const auto& layer : outputLayersInfo) {
        if (strcmp(layer.layerName, "output") == 0) {
            outputLayer = &layer;
            break;
        }
    }
    
    if (!outputLayer) {
        std::cerr << "❌ 找不到output层" << std::endl;
        return false;
    }

    // 输出维度: [1, 25200, 85]
    int numBoxes = outputLayer->inferDims.d[1];      // 25200
    int numAttributes = outputLayer->inferDims.d[2]; // 85
    
    float* data = static_cast<float*>(outputLayer->buffer);
    
    // 获取置信度阈值
    float confThreshold = 0.25f;
    if (detectionParams.numClassesConfigured > 0 && 
        detectionParams.perClassThreshold[0] > 0) {
        confThreshold = detectionParams.perClassThreshold[0];
    }

    int detectionCount = 0;
    
    // 遍历所有预测框
    for (int i = 0; i < numBoxes; i++) {
        float* box = data + i * numAttributes;
        
        // YOLO格式: [cx, cy, w, h, objectness, class0_score, class1_score, ...]
        float cx = box[0];
        float cy = box[1];
        float w = box[2];
        float h = box[3];
        float objectness = box[4];
        
        // 快速过滤低置信度框
        if (objectness < confThreshold) {
            continue;
        }
        
        // 找到最高分类别
        int numClasses = numAttributes - 5;
        int bestClass = 0;
        float bestClassScore = box[5];
        
        for (int c = 1; c < numClasses; c++) {
            float score = box[5 + c];
            if (score > bestClassScore) {
                bestClassScore = score;
                bestClass = c;
            }
        }
        
        // 最终置信度 = objectness * class_score
        float finalConfidence = objectness * bestClassScore;
        
        // 只保留你实际使用的类别（0-4）
        if (bestClass >= detectionParams.numClassesConfigured) {
            continue;
        }
        
        // 检查类别阈值
        if (finalConfidence < detectionParams.perClassThreshold[bestClass]) {
            continue;
        }
        
        // 坐标转换：归一化坐标(0-1) -> 像素坐标
        // YOLO输出是相对于640x640的归一化坐标
        float left = (cx - w * 0.5f) * networkInfo.width;
        float top = (cy - h * 0.5f) * networkInfo.height;
        float width = w * networkInfo.width;
        float height = h * networkInfo.height;
        
        // 边界裁剪
        if (left < 0) left = 0;
        if (top < 0) top = 0;
        if (left + width > networkInfo.width) width = networkInfo.width - left;
        if (top + height > networkInfo.height) height = networkInfo.height - top;
        
        // 过滤无效框
        if (width <= 0 || height <= 0) {
            continue;
        }
        
        // 创建检测对象
        NvDsInferParseObjectInfo obj;
        obj.classId = bestClass;
        obj.detectionConfidence = finalConfidence;
        obj.left = left;
        obj.top = top;
        obj.width = width;
        obj.height = height;
        
        objectList.push_back(obj);
        detectionCount++;
    }
    
    if (detectionCount > 0) {
        std::cout << "✅ YOLO解析: 检测到 " << detectionCount << " 个对象" << std::endl;
    }
    
    return true;
}

// 验证函数签名
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);