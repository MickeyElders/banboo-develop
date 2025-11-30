# Makefile —— 直接覆盖原来的
CFLAGS  = -O3 \
          -I/usr/include/lvgl \
          -I/usr/include/libdrm \
          -I/usr/include/gstreamer-1.0 \
          -I/usr/lib/aarch64-linux-gnu/gstreamer-1.0/include \
          -I/usr/include/glib-2.0 \
          -I/usr/lib/aarch64-linux-gnu/glib-2.0/include

LDFLAGS = -llvgl -llv_drivers \
          -lgstreamer-1.0 -lgstapp-1.0 -lgstvideo-1.0 -lgstbase-1.0 \
          -lgobject-2.0 -lglib-2.0 \
          -ldrm -lgbm -lEGL -lGLESv2 -lpthread

all:
	gcc main.c -o csi_lvgl_demo $(CFLAGS) $(LDFLAGS)

clean:
	rm -f csi_lvgl_demo