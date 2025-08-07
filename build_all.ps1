# 智能切竹机统一编译脚本 (Windows PowerShell版本)
# 支持同时编译C++后端和Flutter前端

param(
    [switch]$Help,
    [switch]$CppOnly,
    [switch]$FlutterOnly,
    [switch]$All,
    [switch]$Debug,
    [switch]$Release,
    [int]$Jobs = 4,
    [string]$Platform = "windows",
    [string]$Target = "desktop"
)

# 颜色定义
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"

# 日志函数
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Red
}

# 显示帮助信息
function Show-Help {
    Write-Host "智能切竹机统一编译脚本 (Windows版本)"
    Write-Host ""
    Write-Host "用法: .\build_all.ps1 [选项]"
    Write-Host ""
    Write-Host "选项:"
    Write-Host "  -Help              显示此帮助信息"
    Write-Host "  -CppOnly           仅编译C++后端"
    Write-Host "  -FlutterOnly       仅编译Flutter前端"
    Write-Host "  -All               编译所有组件 (默认)"
    Write-Host "  -Debug             调试模式编译"
    Write-Host "  -Release           发布模式编译 (默认)"
    Write-Host "  -Jobs N            并行编译任务数 (默认: 4)"
    Write-Host "  -Platform PLAT     目标平台 (windows, linux, android)"
    Write-Host "  -Target TARGET     编译目标 (desktop, embedded, mobile)"
    Write-Host ""
    Write-Host "示例:"
    Write-Host "  .\build_all.ps1                     # 编译所有组件 (发布模式)"
    Write-Host "  .\build_all.ps1 -Debug              # 调试模式编译所有组件"
    Write-Host "  .\build_all.ps1 -CppOnly -Debug     # 仅编译C++后端 (调试模式)"
    Write-Host "  .\build_all.ps1 -FlutterOnly        # 仅编译Flutter前端"
    Write-Host "  .\build_all.ps1 -All -Jobs 8        # 编译所有组件 (8线程)"
}

# 检查依赖
function Test-Dependencies {
    Write-Info "检查编译依赖..."
    
    $missingDeps = @()
    
    # 检查CMake
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        $missingDeps += "cmake"
    }
    
    # 检查Visual Studio或MSBuild
    if (-not (Get-Command msbuild -ErrorAction SilentlyContinue)) {
        if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
            $missingDeps += "Visual Studio Build Tools"
        }
    }
    
    # 检查Git
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        $missingDeps += "git"
    }
    
    # 检查Flutter (如果需要编译前端)
    if ($CompileFlutter) {
        if (-not (Get-Command flutter -ErrorAction SilentlyContinue)) {
            $missingDeps += "flutter"
        }
    }
    
    # 检查OpenCV (Windows版本)
    $opencvPath = "C:\opencv\build\x64\vc16\bin"
    if (-not (Test-Path $opencvPath)) {
        Write-Warning "OpenCV未找到，C++后端编译可能失败"
    }
    
    # 报告缺失的依赖
    if ($missingDeps.Count -gt 0) {
        Write-Error "缺失以下依赖: $($missingDeps -join ', ')"
        Write-Host "请安装缺失的依赖后重试"
        exit 1
    }
    
    Write-Success "依赖检查完成"
}

# 编译C++后端
function Compile-CppBackend {
    Write-Info "开始编译C++后端..."
    
    Push-Location cpp_backend
    
    # 创建构建目录
    if ($BuildType -eq "debug") {
        $buildDir = "build_debug"
        $cmakeBuildType = "Debug"
    } else {
        $buildDir = "build"
        $cmakeBuildType = "Release"
    }
    
    if (-not (Test-Path $buildDir)) {
        New-Item -ItemType Directory -Path $buildDir | Out-Null
    }
    
    Push-Location $buildDir
    
    # 配置CMake
    Write-Info "配置CMake ($cmakeBuildType模式)..."
    cmake .. -DCMAKE_BUILD_TYPE=$cmakeBuildType -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    
    # 编译
    Write-Info "编译C++后端 (使用$Jobs个线程)..."
    cmake --build . --config $cmakeBuildType --parallel $Jobs
    
    # 检查编译结果
    $exePath = ".\bamboo_cut_backend.exe"
    if (Test-Path $exePath) {
        Write-Success "C++后端编译成功"
        Get-ChildItem $exePath | Select-Object Name, Length, LastWriteTime
    } else {
        Write-Error "C++后端编译失败"
        exit 1
    }
    
    Pop-Location
    Pop-Location
}

# 编译Flutter前端
function Compile-FlutterFrontend {
    Write-Info "开始编译Flutter前端..."
    
    Push-Location flutter_frontend
    
    # 获取Flutter依赖
    Write-Info "获取Flutter依赖..."
    flutter pub get
    
    # 检查Flutter环境
    Write-Info "检查Flutter环境..."
    flutter doctor
    
    # 根据目标平台编译
    switch ($Platform) {
        "windows" {
            Write-Info "编译Windows桌面版本..."
            flutter build windows --$BuildType
        }
        "linux" {
            Write-Info "编译Linux桌面版本..."
            flutter build linux --$BuildType
        }
        "android" {
            Write-Info "编译Android版本..."
            flutter build apk --$BuildType
        }
        "web" {
            Write-Info "编译Web版本..."
            flutter build web --$BuildType
        }
        default {
            Write-Info "编译当前平台版本..."
            flutter build --$BuildType
        }
    }
    
    Write-Success "Flutter前端编译成功"
    Pop-Location
}

# 编译嵌入式版本
function Compile-Embedded {
    Write-Info "编译嵌入式版本..."
    
    # 编译C++后端 (嵌入式配置)
    Push-Location cpp_backend
    
    $buildDir = "build_embedded"
    if (-not (Test-Path $buildDir)) {
        New-Item -ItemType Directory -Path $buildDir | Out-Null
    }
    
    Push-Location $buildDir
    
    cmake .. -DCMAKE_BUILD_TYPE=$BuildType -DTARGET_ARCH=aarch64 -DENABLE_EMBEDDED=ON -DENABLE_GUI=OFF -DENABLE_NETWORK=ON
    cmake --build . --config $BuildType --parallel $Jobs
    
    Pop-Location
    Pop-Location
    
    # 编译Flutter嵌入式版本
    Push-Location flutter_frontend
    
    flutter build linux --$BuildType --dart-define=EMBEDDED=true --dart-define=PLATFORM=linux
    
    Pop-Location
}

# 创建部署包
function New-DeploymentPackage {
    Write-Info "创建部署包..."
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $packageName = "bamboo_cut_${BuildType}_$timestamp"
    $packageDir = New-Item -ItemType Directory -Path $packageName -Force
    
    # 复制C++后端
    if ($CompileCpp) {
        $backendDir = Join-Path $packageDir "backend"
        New-Item -ItemType Directory -Path $backendDir -Force | Out-Null
        
        if ($BuildType -eq "debug") {
            $sourceExe = "cpp_backend\build_debug\bamboo_cut_backend.exe"
        } else {
            $sourceExe = "cpp_backend\build\bamboo_cut_backend.exe"
        }
        
        if (Test-Path $sourceExe) {
            Copy-Item $sourceExe $backendDir
        }
        
        # 复制配置文件
        if (Test-Path "cpp_backend\config") {
            Copy-Item "cpp_backend\config\*.yaml" $backendDir -ErrorAction SilentlyContinue
        }
    }
    
    # 复制Flutter前端
    if ($CompileFlutter) {
        $frontendDir = Join-Path $packageDir "frontend"
        New-Item -ItemType Directory -Path $frontendDir -Force | Out-Null
        
        switch ($Platform) {
            "windows" {
                $flutterBuildPath = "flutter_frontend\build\windows\runner\$BuildType"
                if (Test-Path $flutterBuildPath) {
                    Copy-Item "$flutterBuildPath\*" $frontendDir -Recurse
                }
            }
            "linux" {
                $flutterBuildPath = "flutter_frontend\build\linux\x64\$BuildType\bundle"
                if (Test-Path $flutterBuildPath) {
                    Copy-Item "$flutterBuildPath\*" $frontendDir -Recurse
                }
            }
            "android" {
                $apkPath = "flutter_frontend\build\app\outputs\flutter-apk\app-$BuildType.apk"
                if (Test-Path $apkPath) {
                    Copy-Item $apkPath $frontendDir
                }
            }
        }
    }
    
    # 复制文档和脚本
    if (Test-Path "README.md") {
        Copy-Item "README.md" $packageDir
    }
    if (Test-Path "scripts") {
        Copy-Item "scripts\*.sh" $packageDir -ErrorAction SilentlyContinue
    }
    
    # 创建启动脚本
    $startScript = @"
@echo off
REM 智能切竹机启动脚本 (Windows版本)

echo 启动智能切竹机系统...

REM 启动后端服务
if exist ".\backend\bamboo_cut_backend.exe" (
    echo 启动C++后端服务...
    start /B .\backend\bamboo_cut_backend.exe
    set BACKEND_PID=%ERRORLEVEL%
    echo 后端服务PID: %BACKEND_PID%
)

REM 启动前端界面
if exist ".\frontend\bamboo_cut_frontend.exe" (
    echo 启动Flutter前端...
    start /B .\frontend\bamboo_cut_frontend.exe
    set FRONTEND_PID=%ERRORLEVEL%
    echo 前端服务PID: %FRONTEND_PID%
)

echo 系统启动完成
echo 按任意键停止服务
pause

REM 停止服务
taskkill /F /IM bamboo_cut_backend.exe 2>nul
taskkill /F /IM bamboo_cut_frontend.exe 2>nul
"@
    
    $startScript | Out-File -FilePath (Join-Path $packageDir "start.bat") -Encoding ASCII
    
    # 创建压缩包
    $zipPath = "$packageName.zip"
    Compress-Archive -Path $packageDir -DestinationPath $zipPath -Force
    
    Write-Success "部署包创建完成: $zipPath"
    Remove-Item $packageDir -Recurse -Force
}

# 主函数
function Main {
    # 设置默认参数
    $script:CompileCpp = $true
    $script:CompileFlutter = $true
    $script:BuildType = "release"
    
    # 解析参数
    if ($Help) {
        Show-Help
        return
    }
    
    if ($CppOnly) {
        $script:CompileCpp = $true
        $script:CompileFlutter = $false
    }
    
    if ($FlutterOnly) {
        $script:CompileCpp = $false
        $script:CompileFlutter = $true
    }
    
    if ($Debug) {
        $script:BuildType = "debug"
    }
    
    if ($Release) {
        $script:BuildType = "release"
    }
    
    # 显示编译配置
    Write-Info "编译配置:"
    Write-Host "  C++后端: $CompileCpp"
    Write-Host "  Flutter前端: $CompileFlutter"
    Write-Host "  编译模式: $BuildType"
    Write-Host "  并行任务: $Jobs"
    Write-Host "  目标平台: $Platform"
    Write-Host "  目标类型: $Target"
    Write-Host ""
    
    # 检查依赖
    Test-Dependencies
    
    # 记录开始时间
    $startTime = Get-Date
    
    # 根据目标类型选择编译方式
    if ($Target -eq "embedded") {
        Compile-Embedded
    } else {
        # 编译C++后端
        if ($CompileCpp) {
            Compile-CppBackend
        }
        
        # 编译Flutter前端
        if ($CompileFlutter) {
            Compile-FlutterFrontend
        }
    }
    
    # 创建部署包
    New-DeploymentPackage
    
    # 计算编译时间
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    Write-Success "编译完成！总耗时: $([math]::Round($duration, 2))秒"
    
    # 显示结果
    Write-Host ""
    Write-Info "编译结果:"
    if ($CompileCpp) {
        if ($BuildType -eq "debug") {
            $exePath = "cpp_backend\build_debug\bamboo_cut_backend.exe"
        } else {
            $exePath = "cpp_backend\build\bamboo_cut_backend.exe"
        }
        
        if (Test-Path $exePath) {
            Get-ChildItem $exePath | Select-Object Name, Length, LastWriteTime
        } else {
            Write-Host "C++后端文件未找到"
        }
    }
    
    if ($CompileFlutter) {
        switch ($Platform) {
            "windows" {
                $flutterPath = "flutter_frontend\build\windows\runner\$BuildType"
                if (Test-Path $flutterPath) {
                    Get-ChildItem $flutterPath | Select-Object Name, Length, LastWriteTime
                } else {
                    Write-Host "Flutter前端文件未找到"
                }
            }
            "linux" {
                $flutterPath = "flutter_frontend\build\linux\x64\$BuildType\bundle"
                if (Test-Path $flutterPath) {
                    Get-ChildItem $flutterPath | Select-Object Name, Length, LastWriteTime
                } else {
                    Write-Host "Flutter前端文件未找到"
                }
            }
            "android" {
                $apkPath = "flutter_frontend\build\app\outputs\flutter-apk"
                if (Test-Path $apkPath) {
                    Get-ChildItem $apkPath | Select-Object Name, Length, LastWriteTime
                } else {
                    Write-Host "Flutter前端文件未找到"
                }
            }
        }
    }
}

# 运行主函数
Main 