# display_driver.py
import lvgl as lv
import os
import glob

def find_touch_device():
    """
    自动搜索触摸输入设备
    """
    touch_devices = []
    
    # 搜索触摸设备的常见路径
    touch_paths = [
        '/dev/input/event*',
        '/dev/input/touchscreen*',
        '/dev/input/touch*'
    ]
    
    for pattern in touch_paths:
        devices = glob.glob(pattern)
        touch_devices.extend(devices)
    
    # 去重并过滤有效设备
    valid_devices = []
    for device in set(touch_devices):
        if os.path.exists(device) and os.access(device, os.R_OK):
            try:
                # 尝试读取设备信息确认是触摸设备
                with open(device, 'rb') as f:
                    # 简单验证设备可读
                    pass
                valid_devices.append(device)
                print(f"发现触摸设备: {device}")
            except (IOError, OSError):
                continue
    
    # 优先选择event设备
    for device in valid_devices:
        if 'event' in device:
            print(f"选择触摸设备: {device}")
            return device
    
    # 如果没有event设备，返回第一个有效设备
    if valid_devices:
        print(f"选择触摸设备: {valid_devices[0]}")
        return valid_devices[0]
    else:
        print("未检测到触摸设备")
        return None

def init():
    """Initialize display driver for Jetson"""
    
    # For Jetson with framebuffer
    try:
        # Initialize framebuffer driver
        import lv_drivers.linux_fb as fb
        fb.init()
        
        # Create display buffer
        disp_buf1 = lv.disp_draw_buf_t()
        buf1_1 = bytearray(1280 * 720 * 4)  # Adjust for your resolution
        disp_buf1.init(buf1_1, None, len(buf1_1)//4)
        
        # Register display driver
        disp_drv = lv.disp_drv_t()
        disp_drv.init()
        disp_drv.draw_buf = disp_buf1
        disp_drv.flush_cb = fb.flush
        disp_drv.hor_res = 1280
        disp_drv.ver_res = 720
        disp_drv.register()
        
        # Initialize input device (touchscreen/mouse)
        try:
            import lv_drivers.linux_input as input_dev
            input_dev.init()
            
            indev_drv = lv.indev_drv_t()
            indev_drv.init()
            indev_drv.type = lv.INDEV_TYPE.POINTER
            indev_drv.read_cb = input_dev.read
            indev_drv.register()
        except:
            print("Warning: Input device not initialized")
        
    except Exception as e:
        print(f"Display driver initialization failed: {e}")
        # Fallback to SDL for development/testing
        try:
            import SDL
            SDL.init()
            
            # Create SDL display
            disp_buf1 = lv.disp_draw_buf_t()
            buf1_1 = bytearray(1280 * 720 * 4)
            disp_buf1.init(buf1_1, None, len(buf1_1)//4)
            
            disp_drv = lv.disp_drv_t()
            disp_drv.init()
            disp_drv.draw_buf = disp_buf1
            disp_drv.flush_cb = SDL.monitor_flush
            disp_drv.hor_res = 1280
            disp_drv.ver_res = 720
            disp_drv.register()
            
            # Mouse input
            indev_drv = lv.indev_drv_t()
            indev_drv.init()
            indev_drv.type = lv.INDEV_TYPE.POINTER
            indev_drv.read_cb = SDL.mouse_read
            indev_drv.register()
            
        except Exception as e2:
            print(f"SDL fallback also failed: {e2}")
            raise