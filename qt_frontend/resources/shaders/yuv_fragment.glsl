#version 300 es
precision mediump float;

in vec2 vTexCoord;

uniform sampler2D uTextureY;
uniform sampler2D uTextureU;
uniform sampler2D uTextureV;

out vec4 FragColor;

void main()
{
    float y = texture(uTextureY, vTexCoord).r;
    float u = texture(uTextureU, vTexCoord).r - 0.5;
    float v = texture(uTextureV, vTexCoord).r - 0.5;
    
    // YUV to RGB conversion (BT.709)
    float r = y + 1.5748 * v;
    float g = y - 0.1873 * u - 0.4681 * v;
    float b = y + 1.8556 * u;
    
    FragColor = vec4(r, g, b, 1.0);
}