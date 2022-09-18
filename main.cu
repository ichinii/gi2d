// #include "draw.cu"
#include "light_map.cu"
#include "render.cu"
#include "display.cu"

using Args = std::map<std::string, float>;

Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        auto s = std::string(argv[i]);

        auto d = s.find("=");
        if (d == std::string::npos)
            continue;

        auto key = s.substr(0, d);
        auto value = std::stof(s.substr(d+1));
        args[key] = value;
    }
    return args;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    auto args = parse_args(argc, argv);
    args.try_emplace("rays_per_pixel", 32);

    // create resources
    Image* image;
    Image* image_done;
    const auto pixel_count = res.x * res.y;
    // const auto pixel_count = tile_res.x * tile_res.y;
    cudaMallocManaged(&image, pixel_count * sizeof(Image));
    cudaMallocManaged(&image_done, pixel_count * sizeof(Image));

    Tile* tiles;
    Tile* prev_tiles;
    const auto tile_count = tile_res.x * tile_res.y;
    cudaMallocManaged(&tiles, tile_count * sizeof(Tile));
    cudaMallocManaged(&prev_tiles, tile_count * sizeof(Tile));

    cudaDeviceSynchronize();

    // render an image using ray marching
    auto update = [&] (float t) {
        int w = 256;
        int b_tiles = (tile_res.x*tile_res.y-1)/w+1;
        int b_image = (res.x*res.y-1)/w+1;

        std::swap(tiles, prev_tiles);
        cudaDeviceSynchronize();
        pre_process<<<b_tiles, w>>>(tiles);

        cudaDeviceSynchronize();
        light_map<<<b_image, w>>>(tiles, prev_tiles, t);

        cudaDeviceSynchronize();
        post_process<<<b_tiles, w>>>(tiles, prev_tiles);

        // cudaDeviceSynchronize();
        // pre_process<<<b_tiles, w>>>(image);
        cudaDeviceSynchronize();
        render<<<b_image, w>>>(image, tiles, t);

        // for (int i = 0; i < 1; ++i) {
            // cudaDeviceSynchronize();
            // std::swap(image_done, image);
            // blur<<<b_image, w>>>(image_done, image, t);
        // }

        // cudaDeviceSynchronize();
        // render_tiles<<<b_image, w>>>(image, tiles);

        cudaDeviceSynchronize();
        return image;
    };

    // main loop. render and display
    display(update);

    // clean up resources
    cudaFree(image);
    cudaFree(tiles);

    return 0;
}
