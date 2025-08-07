#include <iostream>
#include <opencv2/opencv.hpp>
#include "bamboo_cut/vision/stereo_vision.h"

using namespace bamboo_cut::vision;

int main(int argc, char* argv[]) {
    std::cout << "=== 竹子切割机 - 立体视觉标定工具 ===" << std::endl;
    
    if (argc < 2) {
        std::cout << "用法: " << argv[0] << " <模式>" << std::endl;
        std::cout << "模式:" << std::endl;
        std::cout << "  calibrate  - 进行立体标定" << std::endl;
        std::cout << "  test       - 测试现有标定" << std::endl;
        std::cout << "  disparity  - 实时显示视差图" << std::endl;
        return -1;
    }
    
    std::string mode = argv[1];
    
    // 创建立体视觉系统
    CameraSyncConfig config;
    config.left_device = "/dev/video0";
    config.right_device = "/dev/video1";  // 修复: 从video2改为video1
    config.width = 1280;
    config.height = 720;
    config.fps = 30;
    
    StereoVision stereo_vision(config);
    
    if (!stereo_vision.initialize()) {
        std::cerr << "立体视觉系统初始化失败" << std::endl;
        return -1;
    }
    
    if (mode == "calibrate") {
        std::cout << "\n🎯 开始立体标定..." << std::endl;
        std::cout << "请准备棋盘格标定板 (9x6，方格大小25mm)" << std::endl;
        std::cout << "按 SPACE 捕获标定图像，ESC 退出，ENTER 开始标定" << std::endl;
        
        // 开始标定
        cv::Size board_size(9, 6);
        float square_size = 25.0f; // 25mm
        
        if (!stereo_vision.start_calibration(board_size, square_size)) {
            std::cerr << "无法开始标定" << std::endl;
            return -1;
        }
        
        int captured_frames = 0;
        const int required_frames = 20;
        
        while (captured_frames < required_frames) {
            StereoFrame frame;
            if (!stereo_vision.capture_stereo_frame(frame)) {
                continue;
            }
            
            // 检测标定板
            auto detection = stereo_vision.detect_calibration_pattern(frame.left_image, frame.right_image);
            
            // 显示图像
            cv::Mat display_left = frame.left_image.clone();
            cv::Mat display_right = frame.right_image.clone();
            
            if (detection.left_found) {
                cv::drawChessboardCorners(display_left, board_size, detection.left_corners, true);
            }
            if (detection.right_found) {
                cv::drawChessboardCorners(display_right, board_size, detection.right_corners, true);
            }
            
            // 合并显示
            cv::Mat combined;
            cv::hconcat(display_left, display_right, combined);
            cv::resize(combined, combined, cv::Size(1280, 360));
            
            // 显示状态信息
            std::string status = "已捕获: " + std::to_string(captured_frames) + "/" + std::to_string(required_frames);
            if (detection.left_found && detection.right_found) {
                status += " - 检测成功! 按SPACE捕获";
                cv::putText(combined, status, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            } else {
                status += " - 未检测到标定板";
                cv::putText(combined, status, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            }
            
            cv::imshow("立体标定", combined);
            
            int key = cv::waitKey(30) & 0xFF;
            if (key == 27) { // ESC
                break;
            } else if (key == 32 && detection.left_found && detection.right_found) { // SPACE
                if (stereo_vision.add_calibration_frame(frame.left_image, frame.right_image)) {
                    captured_frames++;
                    std::cout << "✅ 捕获标定帧 " << captured_frames << "/" << required_frames << std::endl;
                }
            } else if (key == 13 && captured_frames >= 10) { // ENTER
                break;
            }
        }
        
        if (captured_frames >= 10) {
            std::cout << "\n🔄 开始标定计算..." << std::endl;
            if (stereo_vision.calibrate_cameras()) {
                std::string save_path = "/opt/bamboo-cut/config/stereo_calibration.xml";
                if (stereo_vision.save_calibration(save_path)) {
                    std::cout << "✅ 标定完成并保存到: " << save_path << std::endl;
                    
                    auto params = stereo_vision.get_calibration_params();
                    std::cout << "📊 标定结果:" << std::endl;
                    std::cout << "  基线距离: " << params.baseline << "mm" << std::endl;
                    std::cout << "  图像尺寸: " << params.image_size.width << "x" << params.image_size.height << std::endl;
                } else {
                    std::cerr << "❌ 保存标定文件失败" << std::endl;
                }
            } else {
                std::cerr << "❌ 标定失败" << std::endl;
            }
        } else {
            std::cout << "❌ 标定帧数不足" << std::endl;
        }
    }
    else if (mode == "test") {
        std::cout << "\n🧪 测试立体标定..." << std::endl;
        
        std::string calib_path = "/opt/bamboo-cut/config/stereo_calibration.xml";
        if (!stereo_vision.load_calibration(calib_path)) {
            std::cerr << "❌ 无法加载标定文件: " << calib_path << std::endl;
            return -1;
        }
        
        auto params = stereo_vision.get_calibration_params();
        std::cout << "✅ 标定文件加载成功" << std::endl;
        std::cout << "📊 标定参数:" << std::endl;
        std::cout << "  基线距离: " << params.baseline << "mm" << std::endl;
        std::cout << "  图像尺寸: " << params.image_size.width << "x" << params.image_size.height << std::endl;
        
        std::cout << "\n📷 实时测试 (按ESC退出)..." << std::endl;
        
        while (true) {
            StereoFrame frame;
            if (!stereo_vision.capture_stereo_frame(frame)) {
                continue;
            }
            
            if (!frame.valid) {
                continue;
            }
            
            // 显示校正后的图像
            cv::Mat rectified_combined;
            cv::hconcat(frame.left_image, frame.right_image, rectified_combined);
            cv::resize(rectified_combined, rectified_combined, cv::Size(1280, 360));
            
            // 绘制极线
            for (int i = 0; i < rectified_combined.rows; i += 32) {
                cv::line(rectified_combined, cv::Point(0, i), cv::Point(rectified_combined.cols, i), 
                        cv::Scalar(0, 255, 0), 1);
            }
            
            cv::putText(rectified_combined, "校正后图像 (绿线应对齐)", cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            cv::imshow("立体校正测试", rectified_combined);
            
            if (cv::waitKey(30) == 27) { // ESC
                break;
            }
        }
    }
    else if (mode == "disparity") {
        std::cout << "\n📐 实时视差图显示..." << std::endl;
        
        std::string calib_path = "/opt/bamboo-cut/config/stereo_calibration.xml";
        if (!stereo_vision.load_calibration(calib_path)) {
            std::cerr << "❌ 无法加载标定文件: " << calib_path << std::endl;
            std::cout << "将使用未校正图像进行视差计算" << std::endl;
        }
        
        std::cout << "📷 实时视差图 (按ESC退出)..." << std::endl;
        
        while (true) {
            StereoFrame frame;
            if (!stereo_vision.capture_stereo_frame(frame)) {
                continue;
            }
            
            if (!frame.valid) {
                continue;
            }
            
            // 显示原图和视差图
            cv::Mat left_small, disparity_vis;
            cv::resize(frame.left_image, left_small, cv::Size(640, 360));
            
            if (!frame.disparity.empty()) {
                disparity_vis = stereo_utils::visualize_disparity(frame.disparity, true);
                cv::resize(disparity_vis, disparity_vis, cv::Size(640, 360));
                
                cv::Mat combined;
                cv::hconcat(left_small, disparity_vis, combined);
                
                cv::putText(combined, "左图像", cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                cv::putText(combined, "视差图", cv::Point(650, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                
                cv::imshow("立体视差", combined);
            } else {
                cv::putText(left_small, "视差计算失败", cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                cv::imshow("立体视差", left_small);
            }
            
            if (cv::waitKey(30) == 27) { // ESC
                break;
            }
        }
    }
    else {
        std::cerr << "未知模式: " << mode << std::endl;
        return -1;
    }
    
    cv::destroyAllWindows();
    std::cout << "\n👋 程序退出" << std::endl;
    
    return 0;
}