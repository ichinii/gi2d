#pragma once

#include "common.cu"

std::map<int, int> key_states;
float mouse_scroll_y = 10.0f;

static GLuint loadShaderFromSourceCode(GLenum type, const char* sourcecode, int length)
{
    GLuint shaderId = glCreateShader(type);

    glShaderSource(shaderId, 1, &sourcecode, &length);
    glCompileShader(shaderId);

    GLint isCompiled = 0;
    glGetShaderiv(shaderId, GL_COMPILE_STATUS, &isCompiled);
    if(isCompiled == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &maxLength);

        auto errorLog = std::make_unique<GLchar[]>(maxLength);
        glGetShaderInfoLog(shaderId, maxLength, &maxLength, &errorLog[0]);

        std::cout << "Error compiling " << std::endl
            << &errorLog[0] << std::endl;
        glDeleteShader(shaderId); // Don't leak the shader.
        return 0;
    }

    return shaderId;
}

static GLuint loadShaderFromFile(GLenum type, const char* filepath)
{
    std::cout << "Loading shader '" << filepath << "'" << std::endl;

    std::ifstream fstream;
    fstream.open(filepath);

    if (!fstream.is_open())
    {
        std::cout << "Unable to open file '" << filepath << "'" << std::endl;
        return 0;
    }

    std::stringstream sstream;
    std::string line;
    while (std::getline(fstream, line))
        sstream << line << '\n';
    line = sstream.str();

    GLuint shaderId = loadShaderFromSourceCode(type, line.c_str(), line.length());
    if (!shaderId)
        std::cout << "...with filepath '" << filepath << "'"; 

    return shaderId;
}

struct shader_load_data_t {
    GLenum type;
    const char* filepath;
};

static GLuint createProgram(std::vector<shader_load_data_t> shader_load_data)
{
    GLuint program;
    program = glCreateProgram();

    std::vector<GLuint> shaders;
    shaders.reserve(shader_load_data.size());
    for (auto& s : shader_load_data) {
        GLuint shader = loadShaderFromFile(s.type, s.filepath);
        shaders.push_back(shader);
        glAttachShader(program, shader);
    }

    glLinkProgram(program);

    for (auto& s : shaders)
        glDeleteShader(s);

    return program;
}

void display(std::function<Image*(float)> update) {
    assert(glfwInit() == GLFW_TRUE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    auto window = glfwCreateWindow(res.x, res.y, "pathtracer", nullptr, nullptr);
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
    glfwMakeContextCurrent(window);
    assert(glewInit() == GLEW_OK);

    glfwSwapInterval(1);
    glClearColor(.2, .1, 0, 1);
    glViewport(0, 0, res.x, res.y);

    // std::this_thread::sleep_for(1s);
    auto display_program = createProgram({
        {GL_VERTEX_SHADER, "vertex.glsl"},
        {GL_FRAGMENT_SHADER, "fragment.glsl"}
    });
    glUseProgram(display_program);

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    enum { vertex_position, vertex_uv };
    GLuint vao;
    GLuint vbos[2];
    glCreateVertexArrays(1, &vao);
    glGenBuffers(2, vbos);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(vertex_position);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[vertex_position]);
    glm::vec2 vertex_positions[] = { {-1, -1}, {1, -1}, {1, 1}, {-1, -1}, {1, 1}, {-1, 1} };
    glBufferData(GL_ARRAY_BUFFER, sizeof (vertex_positions), vertex_positions, GL_STATIC_DRAW);
    glVertexAttribPointer(vertex_position, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(vertex_uv);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[vertex_uv]);
    glm::vec2 vertex_uvs[] = { {0, 0}, {1, 0}, {1, 1}, {0, 0}, {1, 1}, {0, 1} };
    glBufferData(GL_ARRAY_BUFFER, sizeof (vertex_uvs), vertex_uvs, GL_STATIC_DRAW);
    glVertexAttribPointer(vertex_uv, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    glfwSetKeyCallback(window, [] (GLFWwindow *window, int key, int scancode, int action, int mods) {
        if (!mods && key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, true);
        int value = action == GLFW_PRESS ? 1 : (action == GLFW_RELEASE ? -1 : 0);
        if (value != 0)
            key_states.insert_or_assign(key, value);
    });
    auto update_key_states = [] {
        for (auto& [_, v] : key_states) {
            v += sign(v);
        }
    };
    glfwSetScrollCallback(window, [] (GLFWwindow* window, double dx, double dy) {
        mouse_scroll_y = max(3.0f, mouse_scroll_y - dy);
    });

    while (!glfwWindowShouldClose(window)) {
        // handle input
        update_key_states();
        glfwPollEvents();

        // draw the image using ray marching
        Image* image = update(glfwGetTime());
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, res.x, res.y, 0, GL_RGBA, GL_FLOAT, image);
        // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, tile_res.x, tile_res.y, 0, GL_RGBA, GL_FLOAT, image);

        // present the image to screen
        glUseProgram(display_program);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_2D, tex);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glfwSwapBuffers(window);
    }

    glfwTerminate();
}
