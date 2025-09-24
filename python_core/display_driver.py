# display_driver.py
import lvgl as lv

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