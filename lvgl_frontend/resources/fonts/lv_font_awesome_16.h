/*******************************************************************************
 * Size: 16 px
 * Bpp: 1
 * Font Awesome icons
 ******************************************************************************/

#ifndef LV_FONT_AWESOME_16_H
#define LV_FONT_AWESOME_16_H

#include "lvgl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Font Awesome Unicode definitions
#define FA_VIDEO                "\xEF\x80\x82"      // U+F03D (Camera/Video icon)
#define FA_LINK                 "\xEF\x83\x81"      // U+F0C1 (Link icon)
#define FA_CIRCLE               "\xEF\x84\x91"      // U+F111 (Circle icon for status)
#define FA_ROBOT                "\xEF\x95\x84"      // U+F544 (Robot icon)
#define FA_CHART_LINE           "\xEF\x88\x81"      // U+F201 (Chart line icon)
#define FA_EXCLAMATION_TRIANGLE "\xEF\x81\xB1"      // U+F071 (Warning triangle icon)
#define FA_POWER_OFF            "\xEF\x80\x91"      // U+F011 (Power icon)
#define FA_PLAY                 "\xEF\x81\x8B"      // U+F04B (Play icon)
#define FA_PAUSE                "\xEF\x81\x8C"      // U+F04C (Pause icon)
#define FA_STOP                 "\xEF\x81\x8D"      // U+F04D (Stop icon)
#define FA_CHECK                "\xEF\x80\x8C"      // U+F00C (Check mark icon)

// Font declaration
extern const lv_font_t lv_font_awesome_16;

#ifdef __cplusplus
} /*extern "C"*/
#endif

#endif /*LV_FONT_AWESOME_16_H*/