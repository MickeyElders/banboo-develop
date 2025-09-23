/**
 * LVGL配置文件 - 针对Jetson Orin NX优化
 */

#ifndef LV_CONF_H
#define LV_CONF_H

/*====================
   COLOR SETTINGS
 *====================*/

/* Color depth: 1 (1 byte per pixel), 8 (RGB332), 16 (RGB565), 32 (ARGB8888) */
#define LV_COLOR_DEPTH 32

/* Swap the 2 bytes of RGB565 color. Useful if the display has an 8-bit interface (e.g. SPI) */
#define LV_COLOR_16_SWAP 0

/* Enable more complex drawing routines to manage screens transparency.
 * Can be used if the UI is above another layer, e.g. an OSD menu or video player.
 * Requires `LV_COLOR_DEPTH = 32` colors and the screen's `bg_opa` should be set to non LV_OPA_COVER value */
#define LV_COLOR_SCREEN_TRANSP 1

/* Images pixels with this color will not be drawn if they are chroma keyed) */
#define LV_COLOR_CHROMA_KEY lv_color_hex(0x00ff00)         /*Images pixels with this color will not be drawn if they are chroma keyed)*/

/*=========================
   MEMORY SETTINGS
 *=========================*/

/* Size of the memory available for `lv_mem_alloc()` in bytes (>= 2kB) */
#define LV_MEM_SIZE (128U * 1024U)          /* [bytes] */

/* Set an address for the memory pool instead of allocating it as a normal array. Can be in external SRAM too. */
#define LV_MEM_ADR 0     /*0: unused*/

/* Instead of an address give a memory allocator that will be called to get a memory pool for LVGL. E.g. my_malloc */
#if LV_MEM_ADR == 0
    #undef LV_MEM_POOL_INCLUDE
    #undef LV_MEM_POOL_ALLOC
#endif

/*====================
   HAL SETTINGS
 *====================*/

/* Default display refresh period. LVG will redraw changed areas with this period time */
#define LV_DISP_DEF_REFR_PERIOD 16    /* [ms] 60fps */

/* Input device read period in milliseconds */
#define LV_INDEV_DEF_READ_PERIOD 10   /* [ms] */

/* Use a custom tick source that tells the elapsed time in milliseconds.
 * It removes the need to manually update the tick with `lv_tick_inc()`) */
#define LV_TICK_CUSTOM 1
#if LV_TICK_CUSTOM
    #define LV_TICK_CUSTOM_INCLUDE <sys/time.h>         /*Header for the system time function*/
    #define LV_TICK_CUSTOM_SYS_TIME_EXPR (lv_tick_get_ms())    /*Expression evaluating to current system time in ms*/
#endif   /*LV_TICK_CUSTOM*/

/* Default Dot Per Inch. Used to initialize default sizes such as widgets sized, style paddings.
 * (Not so important, you can adjust it to modify default sizes and spaces) */
#define LV_DPI_DEF 130     /*[px/inch]*/

/*=========================
   OPERATING SYSTEM SETTINGS
 *=========================*/

/* Select an operating system to use. Possible options:
 *  - LV_OS_NONE
 *  - LV_OS_PTHREAD
 *  - LV_OS_FREERTOS
 *  - LV_OS_CMSIS_RTOS2
 *  - LV_OS_RTTHREAD
 *  - LV_OS_WINDOWS
 *  - LV_OS_CUSTOM */
#define LV_USE_OS LV_OS_PTHREAD

/*========================
   RENDERING CONFIGURATION
 *========================*/

/* Type of coordinates. Should be `int16_t` (or `int32_t` for extreme cases) */
typedef int16_t lv_coord_t;

/* Maximal horizontal and vertical resolution to support by the library.
 * Leave undefined to let LVGL calculate automatically based on lv_coord_t type */
#undef LV_COORD_MAX  /* Let LVGL auto-calculate to avoid redefinition warning */

/* The target "Dots per inch" (DPI) of the display.
 * Default value is `LV_DPI_DEF` from above.
 * You can overwrite it here or in each display driver.*/
#define LV_DPI_DEF 130

/*=================
   GPU CONFIGURATION  
 *=================*/

/* Use Arm's 2D acceleration library Arm 2D (based on Helium or Neon) for the SW renderer */
#define LV_USE_DRAW_ARM2D_SYNC 0

/* Use STM32's DMA2D (Chrom Art) GPU for the SW renderer.
 * See lv_port_disp_template.c for implementation. */
#define LV_USE_DRAW_DMA2D_SYNC 0

/* Enable experimental VG-Lite GPU support in LVGL. */
#define LV_USE_DRAW_VGLITE 0

/* Enable GPU accelerated rendering with OpenGL ES 2.0.
 * Note that alpha blending and anti aliasing is not supported. */
#define LV_USE_DRAW_OPENGLES 1
#if LV_USE_DRAW_OPENGLES
    /* Set to 1 to use textures as framebuffers (with glTexImage2D)
     * Set to 0 to use renderbuffers (with glRenderbufferStorage)
     * Textures are normally faster */
    #define LV_OPENGLES_FB_TEXTURE 1
#endif

/*===============
   LOG SETTINGS
 *===============*/

/* Enable the log module */
#define LV_USE_LOG 1
#if LV_USE_LOG

    /* How important log should be added:
    * LV_LOG_LEVEL_TRACE       A lot of logs to give detailed information
    * LV_LOG_LEVEL_INFO        Log important events
    * LV_LOG_LEVEL_WARN        Log if something unwanted happened but didn't cause a problem
    * LV_LOG_LEVEL_ERROR       Only critical issues, when the system may fail
    * LV_LOG_LEVEL_USER        Only logs added by the user
    * LV_LOG_LEVEL_NONE        Do not log anything */
    #define LV_LOG_LEVEL LV_LOG_LEVEL_WARN

    /* 1: Print the log with 'printf';
    * 0: User need to register a callback with `lv_log_register_print_cb()` */
    #define LV_LOG_PRINTF 1

    /* 1: Enable print timestamp;
     * 0: Disable print timestamp */
    #define LV_LOG_USE_TIMESTAMP 1

    /* Enable/disable LV_LOG_TRACE in modules that produces a huge number of logs */
    #define LV_LOG_TRACE_MEM        1
    #define LV_LOG_TRACE_TIMER      1
    #define LV_LOG_TRACE_INDEV      1
    #define LV_LOG_TRACE_DISP_REFR  1
    #define LV_LOG_TRACE_EVENT      1
    #define LV_LOG_TRACE_OBJ_CREATE 1
    #define LV_LOG_TRACE_LAYOUT     1
    #define LV_LOG_TRACE_ANIM       1

#endif  /*LV_USE_LOG*/

/*=================
   TIMER SETTINGS
 *=================*/

/* Enable `lv_timer_handler()` to be called in `lv_tick_inc()`. */
#define LV_TICK_HAS_TIMER_HANDLER 1

/*=================
   STDLIB SETTINGS
 *=================*/

#define LV_USE_BUILTIN_MEMCPY 0
#define LV_USE_BUILTIN_STRLEN 0
#define LV_USE_BUILTIN_STRCPY 0
#define LV_USE_BUILTIN_STRNCPY 0

/* Possible values
 * - LV_STDLIB_BUILTIN:     LVGL's built in implementation
 * - LV_STDLIB_CLIB:        Standard C functions, like malloc, strlen, etc
 * - LV_STDLIB_MICROPYTHON: MicroPython implementation
 * - LV_STDLIB_RTTHREAD:    RT-Thread implementation
 * - LV_STDLIB_CUSTOM:      Provide custom implementation */
#define LV_USE_STDLIB LV_STDLIB_CLIB

/*==================
   WIDGET USAGE
 *==================*/

/* Documentation of the widgets: https://docs.lvgl.io/latest/en/widgets/index.html */

#define LV_USE_ANIMIMG    1
#define LV_USE_ARC        1
#define LV_USE_BAR        1
#define LV_USE_BTN        1
#define LV_USE_BTNMATRIX  1
#define LV_USE_CANVAS     1
#define LV_USE_CHECKBOX   1
#define LV_USE_DROPDOWN   1
#define LV_USE_IMG        1
#define LV_USE_LABEL      1
#define LV_USE_LINE       1
#define LV_USE_LIST       1
#define LV_USE_MSGBOX     1
#define LV_USE_ROLLER     1
#define LV_USE_SLIDER     1
#define LV_USE_SPAN       1
#define LV_USE_SPINBOX    1
#define LV_USE_SPINNER    1
#define LV_USE_SWITCH     1
#define LV_USE_TEXTAREA   1
#define LV_USE_TABLE      1
#define LV_USE_TABVIEW    1
#define LV_USE_TILEVIEW   1
#define LV_USE_WIN        1

/*==================
   THEME USAGE
 *==================*/

/* A simple, impressive and very complete theme */
#define LV_USE_THEME_DEFAULT 1
#if LV_USE_THEME_DEFAULT

    /* 0: Light mode; 1: Dark mode */
    #define LV_THEME_DEFAULT_DARK 0

    /* 1: Enable grow on press */
    #define LV_THEME_DEFAULT_GROW 1

    /* Default transition time in [ms] */
    #define LV_THEME_DEFAULT_TRANSITION_TIME 80
#endif /*LV_USE_THEME_DEFAULT*/

/* A very simple theme that is a good starting point for a custom theme */
#define LV_USE_THEME_BASIC 1

/* A theme designed for monochrome displays */
#define LV_USE_THEME_MONO 1

/*==================
   FONT USAGE
 *==================*/

/* Montserrat fonts with various styles and sizes for Latin characters.
 * The italic, bold, and bold italic versions are generated only for the normal font */

/* Demonstrate special features */
#define LV_USE_FONT_MONTSERRAT_8     0
#define LV_USE_FONT_MONTSERRAT_10    0
#define LV_USE_FONT_MONTSERRAT_12    1
#define LV_USE_FONT_MONTSERRAT_14    1
#define LV_USE_FONT_MONTSERRAT_16    1
#define LV_USE_FONT_MONTSERRAT_18    1
#define LV_USE_FONT_MONTSERRAT_20    1
#define LV_USE_FONT_MONTSERRAT_22    1
#define LV_USE_FONT_MONTSERRAT_24    1
#define LV_USE_FONT_MONTSERRAT_26    0
#define LV_USE_FONT_MONTSERRAT_28    0
#define LV_USE_FONT_MONTSERRAT_30    0
#define LV_USE_FONT_MONTSERRAT_32    0
#define LV_USE_FONT_MONTSERRAT_34    0
#define LV_USE_FONT_MONTSERRAT_36    0
#define LV_USE_FONT_MONTSERRAT_38    0
#define LV_USE_FONT_MONTSERRAT_40    0
#define LV_USE_FONT_MONTSERRAT_42    0
#define LV_USE_FONT_MONTSERRAT_44    0
#define LV_USE_FONT_MONTSERRAT_46    0
#define LV_USE_FONT_MONTSERRAT_48    0

/* Demonstrate special features */
#define LV_USE_FONT_MONTSERRAT_12_SUBPX      0
#define LV_USE_FONT_MONTSERRAT_28_COMPRESSED 0  /*bpp = 3*/
#define LV_USE_FONT_DEJAVU_16_PERSIAN_HEBREW 0  /*Hebrew, Arabic, Persian letters and all their forms*/
#define LV_USE_FONT_SIMSUN_16_CJK            1  /*1000 most common CJK radicals*/

/* Disable CJK fonts to avoid linking errors */
#undef LV_FONT_CUSTOM_DECLARE

/* Use default LVGL font instead of custom CJK font */
#define LV_FONT_DEFAULT &lv_font_montserrat_14

/* Enable large font format support for complex fonts */
#define LV_FONT_FMT_TXT_LARGE 1

/* Pixel perfect monospace fonts */
#define LV_USE_FONT_UNSCII_8  0
#define LV_USE_FONT_UNSCII_16 0

/* Optionally declare custom fonts here.
 * You can use these fonts as default font too and they will be available globally.
 * E.g. #define LV_FONT_CUSTOM_DECLARE   LV_FONT_DECLARE(my_font_1) LV_FONT_DECLARE(my_font_2) */
#undef LV_FONT_CUSTOM_DECLARE

/* Enables a custom font in the form of a binary file */
#define LV_USE_FONT_BIN 1

/* Enable drawing placeholders when glyph dsc is not found */
#define LV_USE_FONT_PLACEHOLDER 1

/*===================
   TEXT SETTINGS
 *===================*/

/**
 * Select a character encoding for strings.
 * Your IDE or editor should have the same character encoding
 * - LV_TXT_ENC_UTF8
 * - LV_TXT_ENC_ASCII
 */
#define LV_TXT_ENC LV_TXT_ENC_UTF8

/* Can break (wrap) texts on these chars */
#define LV_TXT_BREAK_CHARS " ,.;:-_"

/* If a word is at least this long, will break wherever "prettiest"
 * To disable, set to a value <= 0 */
#define LV_TXT_LINE_BREAK_LONG_LEN 0

/* Minimum number of characters in a long word to put on a line before a break.
 * Depends on LV_TXT_LINE_BREAK_LONG_LEN. */
#define LV_TXT_LINE_BREAK_LONG_PRE_MIN_LEN 3

/* Minimum number of characters in a long word to put on a line after a break.
 * Depends on LV_TXT_LINE_BREAK_LONG_LEN. */
#define LV_TXT_LINE_BREAK_LONG_POST_MIN_LEN 3

/* The control character to use for signaling text recoloring. */
#define LV_TXT_COLOR_CMD "#"

/* Support bidirectional texts. Allows mixing Left-to-Right and Right-to-Left texts.
 * The direction will be processed according to the Unicode Bidirectional Algorithm:
 * https://www.unicode.org/reports/tr9/*/
#define LV_USE_BIDI 0
#if LV_USE_BIDI
    /* Set the default direction. Supported values:
    * `LV_BASE_DIR_LTR` Left-to-Right
    * `LV_BASE_DIR_RTL` Right-to-Left
    * `LV_BASE_DIR_AUTO` detect texts base direction */
    #define LV_BIDI_BASE_DIR_DEF LV_BASE_DIR_AUTO
#endif

/* Enable Arabic/Persian processing
 * In these languages characters should be replaced with an other form based on their position in the text */
#define LV_USE_ARABIC_PERSIAN_CHARS 0

/*===================
   WIDGETS
 *===================*/

/* Documentation of the widgets: https://docs.lvgl.io/latest/en/widgets/index.html */

#define LV_USE_ARC          1

#define LV_USE_ANIMIMG      1

#define LV_USE_BAR          1

#define LV_USE_BTN          1

#define LV_USE_BTNMATRIX    1

#define LV_USE_CANVAS       1

#define LV_USE_CHECKBOX     1

#define LV_USE_DROPDOWN     1   /*Requires: lv_label*/

#define LV_USE_IMG          1   /*Requires: lv_label*/

#define LV_USE_LABEL        1
#if LV_USE_LABEL
    #define LV_LABEL_TEXT_SELECTION 1   /*Enable selecting text of the label*/
    #define LV_LABEL_LONG_TXT_HINT 1    /*Store some extra info in labels to speed up drawing of very long texts*/
#endif

#define LV_USE_LINE         1

#define LV_USE_LIST         1   /*Requires: lv_btn*/

#define LV_USE_MENU         1

#define LV_USE_METER        1

#define LV_USE_MSGBOX       1   /*Requires: lv_btnmatrix*/

#define LV_USE_ROLLER       1   /*Requires: lv_label*/
#if LV_USE_ROLLER
    #define LV_ROLLER_INF_PAGES 7   /*Number of extra "pages" when the roller is infinite*/
#endif

#define LV_USE_SLIDER       1   /*Requires: lv_bar*/

#define LV_USE_SPAN         1
#if LV_USE_SPAN
    /*A line text can contain maximum num of span descriptor */
    #define LV_SPAN_SNIPPET_STACK_SIZE 64
#endif

#define LV_USE_SPINBOX      1

#define LV_USE_SPINNER      1   /*Requires: lv_arc*/

#define LV_USE_SWITCH       1

#define LV_USE_TEXTAREA     1   /*Requires: lv_label*/
#if LV_USE_TEXTAREA != 0
    #define LV_TEXTAREA_DEF_PWD_SHOW_TIME 1500    /*ms*/
#endif

#define LV_USE_TABLE        1

#define LV_USE_TABVIEW      1   /*Requires: lv_btn, lv_btnmatrix, lv_obj*/

#define LV_USE_TILEVIEW     1

#define LV_USE_WIN          1

/*==================
   LAYOUT SETTINGS
 *==================*/

/* Enable LVGL's Grid Layout system */
#define LV_USE_GRID 1

/* Enable LVGL's Flex Layout system */
#define LV_USE_FLEX 1

/*==================
   OTHERS
 *==================*/

/* 1: Enable API to take snapshot for object */
#define LV_USE_SNAPSHOT 1

/* 1: Enable Monkey test */
#define LV_USE_MONKEY 1

/* 1: Enable grid navigation */
#define LV_USE_GRIDNAV 1

/* 1: Enable lv_obj fragment */
#define LV_USE_FRAGMENT 1

/* 1: Support using images as font in label or span widgets */
#define LV_USE_IMGFONT 1

/* 1: Enable an observer pattern implementation */
#define LV_USE_OBSERVER 1

/* 1: Enable Pinyin input method */
/* Requires: lv_keyboard */
#define LV_USE_IME_PINYIN 1
#if LV_USE_IME_PINYIN
    /* 1: Use default thesaurus      */
    /* If you do not use the default thesaurus, be sure to use `lv_ime_pinyin` after setting the thesaurus */
    #define LV_IME_PINYIN_USE_DEFAULT_DICT 1
    /* Set the maximum number of candidate panels that can be displayed. */
    /* This needs to be adjusted according to the size of the screen. */
    #define LV_IME_PINYIN_CAND_TEXT_NUM 6

    /* Use 9 key input(k9)*/
    #define LV_IME_PINYIN_USE_K9_MODE      1
    #if LV_IME_PINYIN_USE_K9_MODE == 1
        #define LV_IME_PINYIN_K9_CAND_TEXT_NUM 3
    #endif // LV_IME_PINYIN_USE_K9_MODE
#endif

/*==================
   EXAMPLES
 *==================*/

/* Enable the examples to be built with the library */
#define LV_BUILD_EXAMPLES 0

/*==================
   DEMO USAGE
 *==================*/

/* Show some widget. It might be required to increase `LV_MEM_SIZE` */
#define LV_USE_DEMO_WIDGETS 0
#if LV_USE_DEMO_WIDGETS
    #define LV_DEMO_WIDGETS_SLIDESHOW 0
#endif

/* Demonstrate the usage of encoder and keyboard */
#define LV_USE_DEMO_KEYPAD_AND_ENCODER 0

/* Benchmark your system */
#define LV_USE_DEMO_BENCHMARK 0

/* Stress test for LVGL */
#define LV_USE_DEMO_STRESS 0

/* Music player demo */
#define LV_USE_DEMO_MUSIC 0
#if LV_USE_DEMO_MUSIC
    #define LV_DEMO_MUSIC_SQUARE    0
    #define LV_DEMO_MUSIC_LANDSCAPE 0
    #define LV_DEMO_MUSIC_ROUND     0
    #define LV_DEMO_MUSIC_LARGE     0
    #define LV_DEMO_MUSIC_AUTO_PLAY 0
#endif

#endif /*LV_CONF_H*/