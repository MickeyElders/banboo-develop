import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

// C++后端函数签名定义
typedef InitializeInferenceServiceNative = ffi.Int32 Function();
typedef InitializeInferenceServiceDart = int Function();

typedef DetectBambooNative = ffi.Int32 Function(
  ffi.Pointer<ffi.Uint8> image_data,
  ffi.Int32 width,
  ffi.Int32 height,
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> result_data,
  ffi.Pointer<ffi.Int32> result_size,
);
typedef DetectBambooDart = int Function(
  ffi.Pointer<ffi.Uint8> image_data,
  int width,
  int height,
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> result_data,
  ffi.Pointer<ffi.Int32> result_size,
);

typedef GetSystemStatusNative = ffi.Int32 Function(
  ffi.Pointer<ffi.Uint8> status_data,
  ffi.Pointer<ffi.Int32> status_size,
);
typedef GetSystemStatusDart = int Function(
  ffi.Pointer<ffi.Uint8> status_data,
  ffi.Pointer<ffi.Int32> status_size,
);

typedef StartCameraCaptureNative = ffi.Int32 Function();
typedef StartCameraCaptureDart = int Function();

typedef StopCameraCaptureNative = ffi.Int32 Function();
typedef StopCameraCaptureDart = int Function();

typedef GetCameraFrameNative = ffi.Int32 Function(
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> frame_data,
  ffi.Pointer<ffi.Int32> width,
  ffi.Pointer<ffi.Int32> height,
  ffi.Pointer<ffi.Int32> channels,
);
typedef GetCameraFrameDart = int Function(
  ffi.Pointer<ffi.Pointer<ffi.Uint8>> frame_data,
  ffi.Pointer<ffi.Int32> width,
  ffi.Pointer<ffi.Int32> height,
  ffi.Pointer<ffi.Int32> channels,
);

typedef SetCuttingParametersNative = ffi.Int32 Function(
  ffi.Float x_coordinate,
  ffi.Int32 blade_number,
  ffi.Float quality_threshold,
);
typedef SetCuttingParametersDart = int Function(
  double x_coordinate,
  int blade_number,
  double quality_threshold,
);

typedef EmergencyStopNative = ffi.Int32 Function();
typedef EmergencyStopDart = int Function();

typedef ShutdownInferenceServiceNative = ffi.Int32 Function();
typedef ShutdownInferenceServiceDart = int Function();

/// C++后端FFI桥接类
class CppBridge {
  static final ffi.DynamicLibrary _lib = _loadLibrary();
  
  // 函数指针
  static late final InitializeInferenceServiceDart _initializeInferenceService;
  static late final DetectBambooDart _detectBamboo;
  static late final GetSystemStatusDart _getSystemStatus;
  static late final StartCameraCaptureDart _startCameraCapture;
  static late final StopCameraCaptureDart _stopCameraCapture;
  static late final GetCameraFrameDart _getCameraFrame;
  static late final SetCuttingParametersDart _setCuttingParameters;
  static late final EmergencyStopDart _emergencyStop;
  static late final ShutdownInferenceServiceDart _shutdownInferenceService;

  static ffi.DynamicLibrary _loadLibrary() {
    if (Platform.isLinux) {
      return ffi.DynamicLibrary.open('libbamboo_cut_backend.so');
    } else {
      throw UnsupportedError('Unsupported platform');
    }
  }

  static void _initializeFunctions() {
    _initializeInferenceService = _lib
        .lookupFunction<InitializeInferenceServiceNative, InitializeInferenceServiceDart>(
            'initialize_inference_service');
    _detectBamboo = _lib
        .lookupFunction<DetectBambooNative, DetectBambooDart>('detect_bamboo');
    _getSystemStatus = _lib
        .lookupFunction<GetSystemStatusNative, GetSystemStatusDart>('get_system_status');
    _startCameraCapture = _lib
        .lookupFunction<StartCameraCaptureNative, StartCameraCaptureDart>('start_camera_capture');
    _stopCameraCapture = _lib
        .lookupFunction<StopCameraCaptureNative, StopCameraCaptureDart>('stop_camera_capture');
    _getCameraFrame = _lib
        .lookupFunction<GetCameraFrameNative, GetCameraFrameDart>('get_camera_frame');
    _setCuttingParameters = _lib
        .lookupFunction<SetCuttingParametersNative, SetCuttingParametersDart>('set_cutting_parameters');
    _emergencyStop = _lib
        .lookupFunction<EmergencyStopNative, EmergencyStopDart>('emergency_stop');
    _shutdownInferenceService = _lib
        .lookupFunction<ShutdownInferenceServiceNative, ShutdownInferenceServiceDart>('shutdown_inference_service');
  }

  /// 初始化推理服务
  static int initializeInferenceService() {
    _initializeFunctions();
    return _initializeInferenceService();
  }

  /// 检测竹子
  static Map<String, dynamic> detectBamboo(Uint8List imageData, int width, int height) {
    final imagePtr = calloc<ffi.Uint8>(imageData.length);
    final resultPtr = calloc<ffi.Pointer<ffi.Uint8>>();
    final resultSizePtr = calloc<ffi.Int32>();

    try {
      // 复制图像数据到C++内存
      for (int i = 0; i < imageData.length; i++) {
        imagePtr[i] = imageData[i];
      }

      final result = _detectBamboo(imagePtr, width, height, resultPtr, resultSizePtr);
      
      if (result == 0 && resultPtr.value != ffi.nullptr) {
        final resultSize = resultSizePtr.value;
        final resultData = Uint8List(resultSize);
        
        // 复制结果数据
        for (int i = 0; i < resultSize; i++) {
          resultData[i] = resultPtr.value[i];
        }

        // 解析JSON结果
        final resultString = String.fromCharCodes(resultData);
        return Map<String, dynamic>.from(jsonDecode(resultString));
      }
      
      return {'error': 'Detection failed', 'code': result};
    } finally {
      calloc.free(imagePtr);
      calloc.free(resultPtr);
      calloc.free(resultSizePtr);
    }
  }

  /// 获取系统状态
  static Map<String, dynamic> getSystemStatus() {
    final statusPtr = calloc<ffi.Uint8>(1024);
    final statusSizePtr = calloc<ffi.Int32>();

    try {
      final result = _getSystemStatus(statusPtr, statusSizePtr);
      
      if (result == 0) {
        final statusSize = statusSizePtr.value;
        final statusData = Uint8List(statusSize);
        
        for (int i = 0; i < statusSize; i++) {
          statusData[i] = statusPtr[i];
        }

        final statusString = String.fromCharCodes(statusData);
        return Map<String, dynamic>.from(jsonDecode(statusString));
      }
      
      return {'error': 'Failed to get system status', 'code': result};
    } finally {
      calloc.free(statusPtr);
      calloc.free(statusSizePtr);
    }
  }

  /// 开始摄像头捕获
  static int startCameraCapture() {
    return _startCameraCapture();
  }

  /// 停止摄像头捕获
  static int stopCameraCapture() {
    return _stopCameraCapture();
  }

  /// 获取摄像头帧
  static Uint8List? getCameraFrame() {
    final framePtr = calloc<ffi.Pointer<ffi.Uint8>>();
    final widthPtr = calloc<ffi.Int32>();
    final heightPtr = calloc<ffi.Int32>();
    final channelsPtr = calloc<ffi.Int32>();

    try {
      final result = _getCameraFrame(framePtr, widthPtr, heightPtr, channelsPtr);
      
      if (result == 0 && framePtr.value != ffi.nullptr) {
        final width = widthPtr.value;
        final height = heightPtr.value;
        final channels = channelsPtr.value;
        final frameSize = width * height * channels;
        
        final frameData = Uint8List(frameSize);
        
        for (int i = 0; i < frameSize; i++) {
          frameData[i] = framePtr.value[i];
        }
        
        return frameData;
      }
      
      return null;
    } finally {
      calloc.free(framePtr);
      calloc.free(widthPtr);
      calloc.free(heightPtr);
      calloc.free(channelsPtr);
    }
  }

  /// 设置切割参数
  static int setCuttingParameters(double xCoordinate, int bladeNumber, double qualityThreshold) {
    return _setCuttingParameters(xCoordinate, bladeNumber, qualityThreshold);
  }

  /// 紧急停止
  static int emergencyStop() {
    return _emergencyStop();
  }

  /// 关闭推理服务
  static int shutdownInferenceService() {
    return _shutdownInferenceService();
  }
} 