#version 400 core

in vec2 uv;

out vec4 color;

uniform sampler2D tex;

void main()
{
    int w = 8;
    vec3 c = vec3(0);
    vec2 res = vec2(textureSize(tex, 0));
    vec2 px = 1.0f / res;
    vec2 uv2 = floor(uv * res / float(w)) / (res/float(w));

    for (int i = 0; i < w*w; ++i) {
        int x = i % w;
        int y = i / w;
        c += texture(tex, uv2 + vec2(x,y)*px).rgb;
    }
    c /= w*w;

    color = vec4(c, 1);
    // color = vec4(1, 0, 0, 1);
}
