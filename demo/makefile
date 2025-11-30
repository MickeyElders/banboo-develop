# Makefile
CFLAGS  = -O3 -I/usr/include/lvgl -I/usr/include/libdrm
LDFLAGS = -llvgl -llv_drivers -lgstreamer-1.0 -lgstapp-1.0 -lgstvideo-1.0 \
          -ldrm -lgbm -lEGL -lGLESv2 -lpthread

all:
	gcc main.c -o csi_lvgl_demo $(CFLAGS) $(LDFLAGS)

clean:
	rm -f csi_lvgl_demo