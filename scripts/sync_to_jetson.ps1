# 智能切竹机 - Windows到Jetson Nano同步脚本
# PowerShell版本，用于Windows开发环境

param(
    [string]$JetsonIP = "192.168.1.10",
    [string]$JetsonUser = "jetson",
    [string]$LocalPath = ".",
    [string]$RemotePath = "~/bamboo-cutting",
    [switch]$DryRun = $false,
    [switch]$SkipTests = $false
)

# 配置
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# 颜色函数
function Write-ColorOutput($ForegroundColor, $Message) {
    Write-Host $Message -ForegroundColor $ForegroundColor
}

function Write-Info($Message) {
    Write-ColorOutput Green "[INFO] $Message"
}

function Write-Warn($Message) {
    Write-ColorOutput Yellow "[WARN] $Message"
}

function Write-Error($Message) {
    Write-ColorOutput Red "[ERROR] $Message"
}

# 检查依赖工具
function Test-Dependencies {
    Write-Info "检查依赖工具..."
    
    # 检查WSL或者rsync
    $hasRsync = $false
    $hasScp = $false
    
    try {
        $null = Get-Command rsync -ErrorAction Stop
        $hasRsync = $true
        Write-Info "找到rsync工具"
    } catch {
        Write-Warn "未找到rsync，将使用scp备用方案"
    }
    
    try {
        $null = Get-Command scp -ErrorAction Stop
        $hasScp = $true
        Write-Info "找到scp工具"
    } catch {
        Write-Warn "未找到scp工具"
    }
    
    if (-not $hasRsync -and -not $hasScp) {
        Write-Error "需要安装rsync或scp工具。建议安装WSL或Git for Windows"
        throw "Missing required tools"
    }
    
    return @{
        HasRsync = $hasRsync
        HasScp = $hasScp
    }
}

# 测试Jetson连接
function Test-JetsonConnection {
    Write-Info "测试Jetson Nano连接..."
    
    try {
        $result = Test-NetConnection -ComputerName $JetsonIP -Port 22 -WarningAction SilentlyContinue
        if ($result.TcpTestSucceeded) {
            Write-Info "Jetson Nano SSH连接正常"
            return $true
        } else {
            Write-Error "无法连接到Jetson Nano ($JetsonIP:22)"
            return $false
        }
    } catch {
        Write-Error "网络连接测试失败: $_"
        return $false
    }
}

# 创建排除文件列表
function New-ExcludeFile {
    $excludeFile = "sync_exclude.txt"
    
    $excludeContent = @"
# 排除文件和目录
*.pyc
__pycache__/
.git/
.vscode/
*.log
*.tmp
.pytest_cache/
dist/
build/
*.egg-info/
node_modules/
temp/
backup/
*.bak
.DS_Store
Thumbs.db
test_image_*.jpg
calibration_result.jpg
edge_detection_result.jpg
exposure_test_*.jpg
*_test_report.txt
*_test_data.json
jetson_setup_report.txt
.env
"@
    
    $excludeContent | Out-File -FilePath $excludeFile -Encoding UTF8
    Write-Info "创建排除文件列表: $excludeFile"
    return $excludeFile
}

# 使用rsync同步
function Sync-WithRsync {
    param($ExcludeFile)
    
    Write-Info "使用rsync同步文件..."
    
    $rsyncArgs = @(
        "-avz",
        "--progress",
        "--delete",
        "--exclude-from=$ExcludeFile"
    )
    
    if ($DryRun) {
        $rsyncArgs += "--dry-run"
        Write-Warn "干运行模式 - 不会实际传输文件"
    }
    
    $rsyncArgs += "$LocalPath/"
    $rsyncArgs += "${JetsonUser}@${JetsonIP}:$RemotePath/"
    
    Write-Info "执行命令: rsync $($rsyncArgs -join ' ')"
    
    try {
        & rsync @rsyncArgs
        if ($LASTEXITCODE -eq 0) {
            Write-Info "rsync同步成功"
            return $true
        } else {
            Write-Error "rsync同步失败，退出码: $LASTEXITCODE"
            return $false
        }
    } catch {
        Write-Error "rsync执行异常: $_"
        return $false
    }
}

# 使用scp同步（备用方案）
function Sync-WithScp {
    Write-Info "使用scp同步文件..."
    
    # 创建临时压缩包
    $tempZip = "bamboo-cutting-sync.zip"
    
    Write-Info "创建同步包..."
    
    # 排除不需要的文件
    $excludePatterns = @(
        "*.pyc", "__pycache__", ".git", ".vscode", "*.log", "*.tmp",
        ".pytest_cache", "dist", "build", "*.egg-info", "temp", "backup"
    )
    
    # 使用PowerShell压缩
    $filesToCompress = Get-ChildItem -Path $LocalPath -Recurse | Where-Object {
        $file = $_
        $shouldExclude = $false
        foreach ($pattern in $excludePatterns) {
            if ($file.Name -like $pattern -or $file.FullName -like "*\$pattern\*") {
                $shouldExclude = $true
                break
            }
        }
        -not $shouldExclude
    }
    
    if ($DryRun) {
        Write-Warn "干运行模式 - 显示将要同步的文件:"
        $filesToCompress | ForEach-Object { Write-Host "  $($_.FullName)" }
        return $true
    }
    
    # 创建压缩包
    Compress-Archive -Path $LocalPath\* -DestinationPath $tempZip -Force
    
    try {
        # 上传压缩包
        Write-Info "上传文件到Jetson Nano..."
        & scp $tempZip "${JetsonUser}@${JetsonIP}:~/"
        
        if ($LASTEXITCODE -ne 0) {
            throw "scp上传失败"
        }
        
        # 在Jetson上解压
        Write-Info "在Jetson Nano上解压文件..."
        $extractCmd = @"
cd ~ && 
mkdir -p $RemotePath && 
unzip -o $tempZip -d $RemotePath && 
rm $tempZip
"@
        
        & ssh "${JetsonUser}@${JetsonIP}" $extractCmd
        
        if ($LASTEXITCODE -ne 0) {
            throw "远程解压失败"
        }
        
        Write-Info "scp同步成功"
        return $true
        
    } catch {
        Write-Error "scp同步失败: $_"
        return $false
    } finally {
        # 清理临时文件
        if (Test-Path $tempZip) {
            Remove-Item $tempZip -Force
        }
    }
}

# 远程执行命令
function Invoke-RemoteCommand {
    param(
        [string]$Command,
        [string]$Description = "远程命令"
    )
    
    Write-Info "执行$Description..."
    
    try {
        & ssh "${JetsonUser}@${JetsonIP}" $Command
        
        if ($LASTEXITCODE -eq 0) {
            Write-Info "$Description 执行成功"
            return $true
        } else {
            Write-Error "$Description 执行失败，退出码: $LASTEXITCODE"
            return $false
        }
    } catch {
        Write-Error "$Description 执行异常: $_"
        return $false
    }
}

# 远程设置Python环境
function Set-RemotePythonEnv {
    Write-Info "配置Jetson Nano Python环境..."
    
    $setupCmd = @"
cd $RemotePath && 
python3 -m pip install --upgrade pip && 
pip3 install -r requirements.txt
"@
    
    return Invoke-RemoteCommand -Command $setupCmd -Description "Python环境配置"
}

# 运行远程测试
function Invoke-RemoteTests {
    if ($SkipTests) {
        Write-Warn "跳过远程测试"
        return $true
    }
    
    Write-Info "运行Jetson Nano上的测试..."
    
    $testCmd = @"
cd $RemotePath && 
python3 test/test_hardware.py
"@
    
    return Invoke-RemoteCommand -Command $testCmd -Description "硬件测试"
}

# 显示同步状态
function Show-SyncStatus {
    Write-Info "获取同步状态..."
    
    $statusCmd = @"
cd $RemotePath && 
echo "=== 项目目录 ===" && 
ls -la && 
echo "" && 
echo "=== Python版本 ===" && 
python3 --version && 
echo "" && 
echo "=== 磁盘使用 ===" && 
df -h . && 
echo "" && 
echo "=== 最近修改的文件 ===" && 
find . -type f -mtime -1 -exec ls -l {} \;
"@
    
    Invoke-RemoteCommand -Command $statusCmd -Description "状态检查"
}

# 主同步流程
function Start-SyncProcess {
    Write-Info "开始Windows到Jetson Nano同步..."
    Write-Info "目标: ${JetsonUser}@${JetsonIP}:$RemotePath"
    
    # 检查依赖
    $deps = Test-Dependencies
    
    # 测试连接
    if (-not (Test-JetsonConnection)) {
        throw "无法连接到Jetson Nano"
    }
    
    # 创建排除文件
    $excludeFile = New-ExcludeFile
    
    try {
        # 选择同步方法
        $syncSuccess = $false
        
        if ($deps.HasRsync) {
            $syncSuccess = Sync-WithRsync -ExcludeFile $excludeFile
        } elseif ($deps.HasScp) {
            $syncSuccess = Sync-WithScp
        }
        
        if (-not $syncSuccess) {
            throw "文件同步失败"
        }
        
        if (-not $DryRun) {
            # 配置Python环境
            Set-RemotePythonEnv
            
            # 运行测试
            Invoke-RemoteTests
            
            # 显示状态
            Show-SyncStatus
        }
        
        Write-Info "同步流程完成!"
        
    } finally {
        # 清理临时文件
        if (Test-Path $excludeFile) {
            Remove-Item $excludeFile -Force
        }
    }
}

# 显示使用帮助
function Show-Help {
    Write-Host @"
智能切竹机 - Windows到Jetson Nano同步脚本

用法:
    .\sync_to_jetson.ps1 [-JetsonIP <IP>] [-JetsonUser <用户名>] [-DryRun] [-SkipTests]

参数:
    -JetsonIP     Jetson Nano的IP地址 (默认: 192.168.1.10)
    -JetsonUser   Jetson Nano的用户名 (默认: jetson)
    -LocalPath    本地项目路径 (默认: 当前目录)
    -RemotePath   远程目标路径 (默认: ~/bamboo-cutting)
    -DryRun       干运行模式，不实际传输文件
    -SkipTests    跳过远程测试

示例:
    .\sync_to_jetson.ps1
    .\sync_to_jetson.ps1 -JetsonIP 192.168.1.20 -JetsonUser ubuntu
    .\sync_to_jetson.ps1 -DryRun
    .\sync_to_jetson.ps1 -SkipTests

前提条件:
    1. 安装WSL或Git for Windows (提供rsync/ssh/scp)
    2. 配置SSH密钥认证到Jetson Nano
    3. Jetson Nano已配置网络连接

"@
}

# 主程序入口
try {
    if ($args -contains "-h" -or $args -contains "--help") {
        Show-Help
        exit 0
    }
    
    Start-SyncProcess
    
} catch {
    Write-Error "同步失败: $_"
    exit 1
} 