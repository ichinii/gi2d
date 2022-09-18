#pragma once

#include <bits/stdc++.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace glm;

// #define res uvec2(1024, 720)
#define res uvec2(1920, 1080)
#define aspect_ratio (float(res.x) / res.y)
#define tile_size 32u
#define tile_res ((ivec2(res)-1)/int(tile_size)+1)

using uint = unsigned int;
#define pi (glm::pi<float>())

#define max_dist 100.0f
#define min_dist 0.001f
#define max_it 500

using SdfScene = float(*)(vec2, float t);

using Image = vec4;

__global__ struct Tile {
    float e; // emission
    vec3 c; // color
    float hits;
};

__device__ struct Material {
    vec3 a; // albedo
    float e; // emissive
};

__device__ float n21(vec2 s) {
    return fract(9583.243 * sin(dot(vec2(395.194, 739.295), s)));
}

__device__ mat2 rotate(float a) {
    return mat2(
        cos(a), -sin(a),
        sin(a), cos(a)
    );
}

__device__ float attenuate(float l) {
    return 1.0f / (1.0f + l*l);
}

__device__ vec2 normal(SdfScene sdf, vec2 p, float t) {
    float l = sdf(p, t);
    float o = min_dist * 0.3f;
    return normalize(
        l - vec2(
            sdf(p - vec2(o, 0), t),
            sdf(p - vec2(0, o), t)
        )
    );
}

__device__ float sd_min(float a, float b) {
    return a < b ? a : b;
}

__device__ struct Sd {
    float l;
    uint i;
};

__device__ Sd sd_min(uint i, Sd a, float b) {
    return a.l < b ? a : Sd { b, i };
}

__device__ float sdf_plane(vec2 p, vec2 n) {
    return dot(p, n);
}

__device__ float sdf_box(vec2 p, vec2 r) {
    p = abs(p);
    return max(p.x - r.x, p.y - r.y);
}

__device__ float sdf_circle(vec2 p, float r) {
    return length(p) - r;
}

__device__ auto sdf_unlit_scene(vec2 p, float t) {
    auto l = max_dist;
    l = sd_min(l, -sdf_box(p, vec2(0.65, 0.45)));
    l = sd_min(l, sdf_box(p, vec2(0.1, 0.1)));
    l = sd_min(l, sdf_box(p - vec2(0.2), vec2(0.1, 0.1)));
    l = sd_min(l, sdf_box(p - vec2(0, 0.3), vec2(0.1, 0.1)));
    return l;
}

__device__ float sdf_lit_scene(vec2 p, float t) {
    auto l = max_dist;
    l = sd_min(l, sdf_box(p + vec2(0.3, 0.3f * (sin(t) * 0.5f + 0.5f)), vec2(0.1, 0.1)));
    // l = sd_min(l, sdf_circle(p + vec2(0.3, 0.3f * (sin(t) * 0.5f + 0.5f)), 0.1));
    l = sd_min(l, sdf_box(p - vec2(0.2, 0.4), vec2(0.02, 0.02)));
    return l;
}

__device__ float sdf_scene(vec2 p, float t) {
    auto l = max_dist;
    l = sd_min(l, sdf_unlit_scene(p, t));
    l = sd_min(l, sdf_lit_scene(p, t));
    return l;
}

__device__ Material mat_unlit_scene(vec2 p, float t) {
    auto l = Sd { max_dist };
    l = sd_min(0, l, -sdf_box(p, vec2(0.65, 0.45)));
    l = sd_min(1, l, sdf_box(p, vec2(0.1, 0.1)));
    l = sd_min(2, l, sdf_box(p - vec2(0.2), vec2(0.1, 0.1)));
    l = sd_min(3, l, sdf_box(p - vec2(0, 0.3), vec2(0.1, 0.1)));
    Material mat[] {
        { vec3(0.5), 0.0f, },
        { vec3(1, 0, 0), 0.0f, },
        { vec3(0.1, 0.4, 1), 0.0f, },
        { vec3(0.1, 1, 0.1), 0.0f, },
    };
    return mat[l.i];
}

__device__ Material mat_lit_scene(vec2 p, float t) {
    auto l = Sd { max_dist };
    l = sd_min(0, l, sdf_box(p + vec2(0.3, 0.3f * (sin(t) * 0.5f + 0.5f)), vec2(0.1, 0.1)));
    // l = sd_min(0, l, sdf_circle(p + vec2(0.3, 0.3f * (sin(t) * 0.5f + 0.5f)), 0.1));
    l = sd_min(1, l, sdf_box(p - vec2(0.2, 0.4), vec2(0.02, 0.02)));
    Material mat[] {
        { vec3(1), sign(sin(t/2)) * 1.0f, },
        { vec3(0.1, 0.4, 1), 1.0f, },
    };
    return l.l > 0.0f ? Material { vec3(0), 0.0f } : mat[l.i];
}

__device__ Material mat_scene(vec2 p, float t) {
    auto l = Sd { max_dist };
    l = sd_min(0, l, -sdf_box(p, vec2(0.65, 0.45)));
    l = sd_min(1, l, sdf_box(p, vec2(0.1, 0.1)));
    l = sd_min(2, l, sdf_box(p - vec2(0.2), vec2(0.1, 0.1)));
    l = sd_min(3, l, sdf_box(p - vec2(0, 0.3), vec2(0.1, 0.1)));

    l = sd_min(4, l, sdf_box(p + vec2(0.3, 0.3f * (sin(t) * 0.5f + 0.5f)), vec2(0.1, 0.1)));
    // l = sd_min(4, l, sdf_circle(p + vec2(0.3, 0.3f * (sin(t) * 0.5f + 0.5f)), 0.1));
    l = sd_min(5, l, sdf_box(p - vec2(0.2, 0.4), vec2(0.02, 0.02)));
    Material mat[] {
        { vec3(0.5), 0.0f, },
        { vec3(1, 0, 0), 0.0f, },
        { vec3(0.1, 0.4, 1), 0.0f, },
        { vec3(0.1, 1, 0.1), 0.0f, },

        { vec3(1), sign(sin(t/2)) * 1.0f, },
        { vec3(0.1, 0.4, 1), 1.0f, },
    };
    return mat[l.i];
}

__device__ float march(SdfScene sdf, vec2 ro, vec2 rd, float t) {
    auto lo = 0.0f;
    for (int i = 0; i < max_it && lo < max_dist; ++i) {
        float l = max(0.0f, sdf(ro, t));
        ro += l * rd;
        lo += l;

        if (l < min_dist)
            return lo;
    }
    return 0.0f;
}

__device__ vec2 diffuse_rd(vec2 n, vec2 s) {
    float angle = (n21(n + s) - 0.5f) * pi;
    return rotate(angle) * n;
}

// __device__ uvec3 tile_index_to_coord(int i) {
//     return uvec3(i % grid_size, (i / grid_size) % grid_size, i / (grid_size * grid_size));
// }

// __host__ unsigned int tile_coord_to_index(uvec3 v) {
//     return v.x + v.y * grid_size + v.z * grid_size * grid_size;
// }

// std::ostream& operator<< (std::ostream& os, Aabb a) {
//     return os << "(back_left_bot: [" << a.back_left_bot.x << ", " << a.back_left_bot.y << ", " << a.back_left_bot.z << "], front_right_top: [" << a.front_right_top.x << ", " << a.front_right_top.y << ", " << a.front_right_top.z << "])";
// }

// template <typename T>
// void dump(T* a, int n, const char* label) {
//     std::cout << "\t" << label << std::endl;
//     for (int i = 0; i < n; ++i)
//         std::cout << a[i] << ", " << std::endl;
// }
