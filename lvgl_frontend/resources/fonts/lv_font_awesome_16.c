/*******************************************************************************
 * Font Awesome 16px icons for LVGL
 * This is a simplified implementation using basic geometric shapes
 * to represent Font Awesome icons without requiring the actual font file
 ******************************************************************************/

#include "lv_font_awesome_16.h"

/*-----------------
 *    BITMAPS
 *----------------*/

/* Video icon bitmap (simplified camera icon) */
static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap_video[] = {
    0xFF, 0xFF, 0x80, 0x01, 0x80, 0x01, 0x9F, 0xF9, 0x90, 0x09, 0x90, 0x09,
    0x9F, 0xF9, 0x80, 0x01, 0x80, 0x01, 0xFF, 0xFF
};

/* Link icon bitmap (simplified chain link) */
static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap_link[] = {
    0x0F, 0x00, 0x1F, 0x80, 0x10, 0x80, 0x10, 0x80, 0x1F, 0x80, 0x0F, 0x00,
    0x00, 0xF0, 0x01, 0xF8, 0x01, 0x08, 0x01, 0x08, 0x01, 0xF8, 0x00, 0xF0
};

/* Circle icon bitmap */
static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap_circle[] = {
    0x07, 0xE0, 0x1F, 0xF8, 0x3F, 0xFC, 0x7F, 0xFE, 0x7F, 0xFE, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0x7F, 0xFE, 0x7F, 0xFE, 0x3F, 0xFC, 0x1F, 0xF8,
    0x07, 0xE0
};

/* Robot icon bitmap (simplified robot head) */
static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap_robot[] = {
    0x3F, 0xFC, 0x20, 0x04, 0x2F, 0xF4, 0x28, 0x14, 0x28, 0x14, 0x2F, 0xF4,
    0x20, 0x04, 0x27, 0xE4, 0x24, 0x24, 0x3F, 0xFC
};

/* Chart line icon bitmap */
static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap_chart[] = {
    0x00, 0x01, 0x00, 0x03, 0x00, 0x06, 0x00, 0x0C, 0x20, 0x18, 0x60, 0x30,
    0xE0, 0x60, 0xC0, 0xC0, 0x81, 0x80, 0x03, 0x00, 0x06, 0x00, 0xFF, 0xFF
};

/* Warning triangle icon bitmap */
static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap_warning[] = {
    0x01, 0x80, 0x03, 0xC0, 0x07, 0xE0, 0x0E, 0x70, 0x1C, 0x38, 0x39, 0x9C,
    0x73, 0xCE, 0x67, 0xE6, 0xCF, 0xF3, 0x9F, 0xF9, 0x3F, 0xFC, 0x7F, 0xFE,
    0xFF, 0xFF
};

/* Power icon bitmap */
static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap_power[] = {
    0x01, 0x80, 0x01, 0x80, 0x71, 0x8E, 0xF9, 0x9F, 0xF9, 0x9F, 0xF9, 0x9F,
    0xF9, 0x9F, 0xF9, 0x9F, 0x71, 0x8E, 0x01, 0x80, 0x01, 0x80
};

/* Play icon bitmap */
static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap_play[] = {
    0x18, 0x00, 0x1E, 0x00, 0x1F, 0x80, 0x1F, 0xE0, 0x1F, 0xF8, 0x1F, 0xFE,
    0x1F, 0xF8, 0x1F, 0xE0, 0x1F, 0x80, 0x1E, 0x00, 0x18, 0x00
};

/* Pause icon bitmap */
static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap_pause[] = {
    0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C,
    0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C, 0x3C
};

/* Stop icon bitmap */
static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap_stop[] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

/* Check icon bitmap */
static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap_check[] = {
    0x00, 0x03, 0x00, 0x07, 0x00, 0x0E, 0x00, 0x1C, 0x00, 0x38, 0x30, 0x70,
    0x78, 0xE0, 0x7D, 0xC0, 0x3F, 0x80, 0x1F, 0x00, 0x0E, 0x00, 0x04, 0x00
};

/*-----------------
 *    GLYPH DSC
 *----------------*/

static const lv_font_fmt_txt_glyph_dsc_t glyph_dsc[] = {
    {.bitmap_index = 0, .adv_w = 192, .box_w = 12, .box_h = 10, .ofs_x = 0, .ofs_y = 0},  /* VIDEO */
    {.bitmap_index = 15, .adv_w = 192, .box_w = 12, .box_h = 12, .ofs_x = 0, .ofs_y = 0}, /* LINK */
    {.bitmap_index = 33, .adv_w = 192, .box_w = 13, .box_h = 13, .ofs_x = 0, .ofs_y = 0}, /* CIRCLE */
    {.bitmap_index = 55, .adv_w = 192, .box_w = 12, .box_h = 10, .ofs_x = 0, .ofs_y = 0}, /* ROBOT */
    {.bitmap_index = 70, .adv_w = 192, .box_w = 12, .box_h = 12, .ofs_x = 0, .ofs_y = 0}, /* CHART */
    {.bitmap_index = 88, .adv_w = 192, .box_w = 13, .box_h = 13, .ofs_x = 0, .ofs_y = 0}, /* WARNING */
    {.bitmap_index = 110, .adv_w = 192, .box_w = 11, .box_h = 11, .ofs_x = 0, .ofs_y = 0}, /* POWER */
    {.bitmap_index = 126, .adv_w = 192, .box_w = 11, .box_h = 11, .ofs_x = 0, .ofs_y = 0}, /* PLAY */
    {.bitmap_index = 142, .adv_w = 192, .box_w = 10, .box_h = 10, .ofs_x = 0, .ofs_y = 0}, /* PAUSE */
    {.bitmap_index = 155, .adv_w = 192, .box_w = 10, .box_h = 10, .ofs_x = 0, .ofs_y = 0}, /* STOP */
    {.bitmap_index = 168, .adv_w = 192, .box_w = 12, .box_h = 12, .ofs_x = 0, .ofs_y = 0}  /* CHECK */
};

/*-----------------
 *    CHARACTER MAPPING
 *----------------*/

static const uint16_t unicode_list[] = {
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0
};

/*-----------------
 *    FONT
 *----------------*/

static const lv_font_fmt_txt_dsc_t font_dsc = {
    .glyph_bitmap = (const uint8_t*)glyph_bitmap_video,
    .glyph_dsc = glyph_dsc,
    .cmaps = NULL,
    .kern_dsc = NULL,
    .kern_scale = 0,
    .cmap_num = 0,
    .bpp = 1,
    .kern_classes = 0,
    .bitmap_format = 0,
};

const lv_font_t lv_font_awesome_16 = {
    .get_glyph_dsc = lv_font_get_glyph_dsc_fmt_txt,
    .get_glyph_bitmap = lv_font_get_bitmap_fmt_txt,
    .line_height = 16,
    .base_line = 0,
    .subpx = LV_FONT_SUBPX_NONE,
    .underline_position = -2,
    .underline_thickness = 1,
    .dsc = &font_dsc
};
