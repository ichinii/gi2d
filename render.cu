#pragma once

#include "common.cu"

__device__ vec3 trace(Tile* tiles, vec2 ro, vec2 rd, vec3 c, float t) {
    // ro += rd * min_dist;
    const int bounces = 1;
    vec3 ct = vec3(0);
    float lo = 0.0f;
    float hits = 0.0f;
    float hits2 = 0.0f;
    float e = 0.0f;
    vec3 a = vec3(1);
    for (int i = 0; i < bounces; ++i) {
        float l = march(sdf_scene, ro, rd, t);
        lo += l;
        ro = ro + rd * l;
        vec2 n = normal(sdf_scene, ro, t);
        vec2 offset = n * min_dist * 2.0f;
        auto tile_index = pos_to_tile_index(ro - offset);
        auto mat = mat_scene(ro, t);
        c = c * attenuate(l);
        a *= mat.a;

        if (mat.e > 0.0f) {
            e += mat.e;
            ++hits;
            break;
        } else {
            ct = ct * mat.a + tiles[tile_index].c * attenuate(lo);
            ++hits2;
        }

        ro += offset;
        rd = diffuse_rd(n, ro + t);
        // rd = diffuse_rd(n, vec2(t));
    }
    // hits = max(1.0f, hits);
    // hits2 = max(1.0f, hits2);
    // return (c*a*e/hits + ct/hits2);
    if (0.0f < hits+hits2) {
        return (c*a*e+ct)/(hits+hits2);
    }
    return vec3(0.0f);
}

__device__ const vec2 BlueNoiseInDisk[64] {
    vec2(0.478712,0.875764),
    vec2(-0.337956,-0.793959),
    vec2(-0.955259,-0.028164),
    vec2(0.864527,0.325689),
    vec2(0.209342,-0.395657),
    vec2(-0.106779,0.672585),
    vec2(0.156213,0.235113),
    vec2(-0.413644,-0.082856),
    vec2(-0.415667,0.323909),
    vec2(0.141896,-0.939980),
    vec2(0.954932,-0.182516),
    vec2(-0.766184,0.410799),
    vec2(-0.434912,-0.458845),
    vec2(0.415242,-0.078724),
    vec2(0.728335,-0.491777),
    vec2(-0.058086,-0.066401),
    vec2(0.202990,0.686837),
    vec2(-0.808362,-0.556402),
    vec2(0.507386,-0.640839),
    vec2(-0.723494,-0.229240),
    vec2(0.489740,0.317826),
    vec2(-0.622663,0.765301),
    vec2(-0.010640,0.929347),
    vec2(0.663146,0.647618),
    vec2(-0.096674,-0.413835),
    vec2(0.525945,-0.321063),
    vec2(-0.122533,0.366019),
    vec2(0.195235,-0.687983),
    vec2(-0.563203,0.098748),
    vec2(0.418563,0.561335),
    vec2(-0.378595,0.800367),
    vec2(0.826922,0.001024),
    vec2(-0.085372,-0.766651),
    vec2(-0.921920,0.183673),
    vec2(-0.590008,-0.721799),
    vec2(0.167751,-0.164393),
    vec2(0.032961,-0.562530),
    vec2(0.632900,-0.107059),
    vec2(-0.464080,0.569669),
    vec2(-0.173676,-0.958758),
    vec2(-0.242648,-0.234303),
    vec2(-0.275362,0.157163),
    vec2(0.382295,-0.795131),
    vec2(0.562955,0.115562),
    vec2(0.190586,0.470121),
    vec2(0.770764,-0.297576),
    vec2(0.237281,0.931050),
    vec2(-0.666642,-0.455871),
    vec2(-0.905649,-0.298379),
    vec2(0.339520,0.157829),
    vec2(0.701438,-0.704100),
    vec2(-0.062758,0.160346),
    vec2(-0.220674,0.957141),
    vec2(0.642692,0.432706),
    vec2(-0.773390,-0.015272),
    vec2(-0.671467,0.246880),
    vec2(0.158051,0.062859),
    vec2(0.806009,0.527232),
    vec2(-0.057620,-0.247071),
    vec2(0.333436,-0.516710),
    vec2(-0.550658,-0.315773),
    vec2(-0.652078,0.589846),
    vec2(0.008818,0.530556),
    vec2(-0.210004,0.519896) 
};

__global__ void render(Image* image, Tile* tiles, float t) {
    uint gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (res.x*res.y <= gtid)
        return;

    uvec2 uv = uvec2(gtid % res.x, gtid / res.x);
    vec2 ro = (vec2(uv) - vec2(res) * 0.5f) / float(res.y);

    // float offset = n21(ro+t);
    // float offset = n21(ro);
    // float offset = n21(floor(vec2(uv) / vec2(tile_size)) + t*0.1f);
    // float offset = (uv.x + uv.y*res.x)*(pi*2.0f)/(tile_size*tile_size);
    // float offset = n21(vec2(uv.x + uv.y*res.x));
    // float offset = uv.x / float(tile_size);
    // float offset = 0.0f;
    // float offset = n21(vec2(t));
    // float offset = n21(vec2(uv.x%8, uv.y%8));
    int ray_count = 4;

    auto mat = mat_scene(ro, t);
    float l = sdf_unlit_scene(ro, t);
    vec3 c = vec3(0);
    if (0.0f < l) {
        for (int i = 0; i < ray_count; ++i) {
            // float a = (offset + float(i) / ray_count) * pi * 2.0f;
            // vec2 rd = vec2(cos(a), sin(a));
            uint tile_index = uv.x / tile_size + uv.y / tile_size * tile_res.x;
            vec2 rd = normalize(
                rotate(
                    BlueNoiseInDisk[
                        (i) % 64
                    ].x
                    * pi * 2.0f
                )
                * BlueNoiseInDisk[uv.x%8 + (uv.y%8)*8]
            );
            c += trace(tiles, ro, rd, vec3(1), t);
        }
        c /= ray_count;
    } else {
        // uint tile_index = uv.x / tile_size + uv.y / tile_size * tile_res.x;
        // c = tiles[tile_index].c;
        c = mat.a * 0.1f;
    }

    // gamma correction
    c = pow(c, vec3(1.0 / 2.2));

    // uint tile_index = uv.x / tile_size + uv.y / tile_size * tile_res.x;
    // atomicAdd(&image[tile_index].r, c.r/(tile_size*tile_size));
    // atomicAdd(&image[tile_index].g, c.g/(tile_size*tile_size));
    // atomicAdd(&image[tile_index].b, c.b/(tile_size*tile_size));
    image[gtid] = mix(vec4(c, 1.0f), image[gtid], 0.0f);
}

__global__ void pre_process(Image* image) {
    uint gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < tile_res.x * tile_res.y) {
        image[gtid] = vec4(0, 0, 0, 1);
    }
}

// __device__ vec3 kernel(Image* image, ivec2 uv, vec2 ro, float t) {
//     const float e = 1.0f / res.y;
//     vec3 acc = vec3(0);
//     float count = 0.0f;
//     int r = 5;
//     for (int i = 0; i < r*r; ++i) {
//         ivec2 offset = ivec2(i%r, i/r) - r/2;
//         float s = step(0.0f, sdf_scene(ro + vec2(offset) * e, t));
//         float l = s / (length(vec2(offset))+1.0f);
//         ivec2 luv = glm::clamp(uv + offset, ivec2(0), ivec2(res)-1);
//         int pixel_index = luv.x + luv.y * res.x;
//         acc += l * vec3(image[pixel_index]);
//         count += l;
//     }
//     return acc / count;
// }

// __global__ void blur(Image* image_dst, Image* image_src, float t) {
//     uint gtid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (res.x*res.y <= gtid)
//         return;

//     uvec2 uv = uvec2(gtid % res.x, gtid / res.x);
//     vec2 ro = (vec2(uv) - vec2(res) * 0.5f) / float(res.y);

//     image_dst[gtid] = vec4(kernel(image_src, uv, ro, t), image_src[gtid].a);
// }

// __device__ void scan(Image* a, Image* b) {
//     int tid = threadIdx.x;
//     __shared__ Image c[32];
//     c[tid] = a[tid];
//     __syncwarp;
//     for (int i = 1; i < radix_threads; i*=2) {
//         Image ai = i <= tid ? a[tid - i] : 0;
//         a[tid] = a[tid] + ai;
//         __syncwarp;
//     }
//     if (tid == 0)
//         b[blockIdx.x] = c[tid];
// }

// __global__ void tiled_image(Image* image_dst, Image* image_src) {
//     int tid = threadIdx.x;
// }
