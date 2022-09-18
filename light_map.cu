#pragma once

#include "common.cu"

__device__ uint pos_to_tile_index(vec2 p) {
    p = clamp(p / vec2(aspect_ratio, 1.0f) + 0.5f, vec2(0), vec2(0.999));
    uvec2 tile_coord = uvec2(p * vec2(tile_res));
    uint tile_index = tile_coord.x + tile_coord.y * tile_res.x;
    return tile_index;
}

__device__ void trace(Tile* tiles, Tile* prev_tiles, vec2 ro, vec2 rd, vec3 c, float t) {
    const int bounces = 8;
    float lo = 0.0f;
    int hit = c == vec3(0) ? 0 : 1;
    vec3 cp = vec3(0);
    vec3 a = vec3(1);
    for (int i = 0; i < bounces; ++i) {
        float l = march(sdf_unlit_scene, ro, rd, t);
        lo += l;
        ro = ro + rd * l;
        vec2 n = normal(sdf_unlit_scene, ro, t);
        vec2 offset = n * min_dist * 2.0f;
        auto tile_index = pos_to_tile_index(ro - offset);
        // vec3 cc = prev_tiles[tile_index].c;

        auto mat = mat_unlit_scene(ro, t);
        // float surface = tile_size;
        // c *= attenuate(l) * (mat.a + prev_tiles[tile_index].c / float(prev_tiles[tile_index].hits+1));
        a *= mat.a;
        // c *= mat.a;
        // c = mix(c, cp, vec3(0.0f));
        cp *= attenuate(l);
        cp *= mat.a;
        // vec3 cc = mix(c, cp, vec3(0.9f));
        vec3 cc = c * a * attenuate(lo) * (i+1.0f);
        cc += cp;
        // cc = mix(prev_tiles[tile_index].c, cc, 0.8f);

        atomicAdd(&tiles[tile_index].c.r, cc.r);
        atomicAdd(&tiles[tile_index].c.g, cc.g);
        atomicAdd(&tiles[tile_index].c.b, cc.b);
        // atomicAdd(&tiles[tile_index].hits, hit + i);
        atomicAdd(&tiles[tile_index].hits, 1);

        ro += offset;
        rd = diffuse_rd(n, ro + t);
        // rd = diffuse_rd(n, vec2(t));
        // rd = normalize(vec2(n21(n) * n.x, n21(n+2.0f) * n.y));

        cp += prev_tiles[tile_index].c;
        cp /= 2.0f;
    }
}

__global__ void light_map(Tile* tiles, Tile* prev_tiles, float t) {
    uint gtid = threadIdx.x + blockIdx.x * blockDim.x;

    uvec2 uv = uvec2(gtid % res.x, gtid / res.x);
    vec2 ro = (vec2(uv) - vec2(res) * 0.5f) / float(res.y);

    auto mat = mat_lit_scene(ro, t);
    float l = sdf_unlit_scene(ro, t);
    if (min_dist < l) {
        float offset = n21(ro + t);
        float ray_count = 3.0f;
        for (float i = 0.0f; i < ray_count; ++i) {
            float a = (offset + i / ray_count) * pi * 2.0f;
            vec2 rd = vec2(cos(a), sin(a));
            vec3 c = vec3(0);
            if (0.0f < mat.e) {
                c = mat.a * mat.e;
                trace(tiles, prev_tiles, ro, rd, c, t);
            }
        }
    }
}

__global__ void pre_process(Tile* tiles) {
    uint gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < tile_res.x * tile_res.y) {
        tiles[gtid].c = vec3(0);
        tiles[gtid].hits = 0;
    }
}

__global__ void post_process(Tile* tiles, Tile* prev_tiles) {
    uint gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < tile_res.x * tile_res.y) {
        if (0 < tiles[gtid].hits) {
            tiles[gtid].c = tiles[gtid].c / float(tiles[gtid].hits);
            tiles[gtid].c = mix(tiles[gtid].c, prev_tiles[gtid].c, vec3(0.9f));
            tiles[gtid].c = clamp(tiles[gtid].c, vec3(0), vec3(1));
            // tiles[gtid].c = tiles[gtid].c / 5.0f;
        }
    }
}

__global__ void render_tiles(Image* image, Tile* tiles) {
    uint gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < res.x * res.y) {
        uvec2 uv = uvec2(gtid % res.x, gtid / res.x);
        uint tile_index = uv.x / tile_size + uv.y / tile_size * tile_res.x;
        image[gtid] = vec4(tiles[tile_index].c, 1.0f);

        // denoise
        // image[gtid] = vec4(pow(vec3(image[gtid]), vec3(2.2f)), image[gtid].a);

        // gamma correct
        // image[gtid] = vec4(pow(vec3(image[gtid]), vec3(1.0f/2.2f)), image[gtid].a);

        // vec2 ro = (vec2(uv) - vec2(res) * 0.5f) / float(res.y);
        // vec2 n = normal(sdf_unlit_scene, ro);
        // image[gtid] = vec4(n, smoothstep(0.003f, 0.0f, sdf_scene(ro)), 1.0f);
        // image[gtid] = vec4(tile_index / float(tile_res.x*tile_res.y), 0, 0, 1.0f);
    }
}
