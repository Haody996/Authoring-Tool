// Dear ImGui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation, etc.)
// If you are new to Dear ImGui, read documentation from the docs/ folder + read the top of imgui.cpp.
// Read online: https://github.com/ocornut/imgui/tree/master/docs

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include <direct.h>
#include <Magick++.h> 
#include <iostream> 
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/utility.hpp>


# define M_PI           3.14159265358979323846  /* pi */
using namespace cv;

using namespace std;
using namespace Magick;


#define GL_CLAMP_TO_EDGE 0x812F

std::string fixedLength(int value, int digits);
int Gradient(int, int);
int PositionalGuide(int, int, string);

void EdgeGuide(string source, string distination);
void createNewFrame(cv::Mat& frame, const cv::Mat& flow, float shift, cv::Mat& next);
int Difference(String f1, String f2, int px, int py, int qx, int qy);
void TemporalGuide(int f, string);
int Difference(Mat f1, Mat f2, int px, int py, int qx, int qy);
int Stylize(string targetFolder, string keyframeFolder, string keyframeLocation);

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Simple helper function to load an image into a OpenGL texture with common settings
bool LoadTextureFromFile(const char* filename, GLuint* out_texture, int* out_width, int* out_height)
{
    // Load from file
    int image_width = 0;
    int image_height = 0;
    unsigned char* image_data = stbi_load(filename, &image_width, &image_height, NULL, 4);
    if (image_data == NULL)
        return false;

    // Create a OpenGL texture identifier
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    // Upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);
    stbi_image_free(image_data);

    *out_texture = image_texture;
    *out_width = image_width;
    *out_height = image_height;

    return true;
}

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif



static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int argc, char** argv)
{
    InitializeMagick(*argv);
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL3 example", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);


    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);


    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();


        //  Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Select your video sequence");                          

            static char buf1[64] = ".\\resources\\target\\"; ImGui::InputText("Paste your video sequence directory here", buf1, 64);
            static char buf2[64] = ".\\keyframe_guides\\"; ImGui::InputText("Paste your video keyframe folder directory here", buf2, 64);
            static char buf3[64] = ".\\keyframe_guides\\keyframe.jpg"; ImGui::InputText("Paste your keyframe file location here", buf3, 64);

            String targetFolder = buf1;
            String keyframeFolder = buf2;
            String keyframeLocation = buf3;

            int my_image_width = 0;
            int my_image_height = 0;
            GLuint my_image_texture = 0;
            
            bool ret = LoadTextureFromFile(buf3, &my_image_texture, &my_image_width, &my_image_height);
            

            //std::cout << buf2 << std::endl;

            ImGui::Text("Press button to start video stylization");               // Display some text (you can use a format strings too)
            //ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
            //ImGui::Checkbox("Another Window", &show_another_window);

            //ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            //ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Start")) {

                Stylize(targetFolder,keyframeFolder,keyframeLocation);

                show_another_window = true;
            }                           
            ImGui::Text("Completed frames: %d", 1);
            ImGui::Text("Your Keyframe:", 1);
            if (ret) {
                ImGui::Image((void*)(intptr_t)my_image_texture, ImVec2(my_image_width, my_image_height));
            }
            
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window)
        {
            ImGui::Begin("Status", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Stylization Completed");
            if (ImGui::Button("Done"))
                show_another_window = false;
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}



int Stylize(string targetFolder,string keyframeFolder, string keyframeLocation)
{
    

    // Construct the image object
    Image target;

    Mat target_000;
    target_000 = imread(targetFolder + "000.jpg");
    int width = target_000.size().width;
    int height = target_000.size().height;

    //weights for error function
    double lambda_col = 7;
    double lambda_pos = 1;
    double lambda_edge = 2;
    double lambda_temp = 0.5;


    //base for positional guide
    target.read(targetFolder + "000.jpg");
    unsigned int w = target.size().width();
    unsigned int h = target.size().height();
    Gradient(w, h);

    //get any positional guide, e.g. between keyframe 0 and target frame 10

    try{
        for (int i = 0; i < 100; i++) {
            string s = fixedLength(i, 3);
            EdgeGuide(targetFolder + s + ".jpg", ".\\edges\\" + s + ".jpg");

        }

    }
    catch (Magick::Exception & error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }


    for (int i = 1; i < 100; i++) {
        PositionalGuide(0, i, targetFolder);
    }
    


    Image key, key_origin;
    key.read(keyframeLocation);
    key_origin.read(targetFolder + "000.jpg");

    Mat col_origin = imread(keyframeFolder + "origin.jpg");
    Mat pos_origin = imread(keyframeFolder + "gradient.jpg");
    Mat edge_origin = imread(keyframeFolder + "edge.jpg");
    Mat keyframe = imread(keyframeLocation);


    for (int i = 1; i < 100; i++) {
        Mat prev = imread(".\\result\\" + fixedLength(i - 1, 3) + ".jpg");



        TemporalGuide(i,targetFolder);
        Mat temporal = imread(".\\temporal\\" + fixedLength(i, 3) + ".jpg");


        //for each pixel in target in scan line order

        int target_frame = i;

        Mat col_target = imread(targetFolder + fixedLength(target_frame, 3) + ".jpg");
        Mat pos_target = imread(".\\positional\\" + fixedLength(0, 3) + "_" + fixedLength(target_frame, 3) + ".jpg");
        Mat edge_target = imread(".\\edges\\" + fixedLength(target_frame, 3) + ".jpg");

        for (int qy = 0; qy < height; qy++) {
            for (int qx = 0; qx < width; qx++) {

                //find in source keyframe

                int minE = 10000000000000;
                int temppy = qy, temppx = qx;

                for (int py = qy - 20; py < qy + 20; py++) {
                    for (int px = qx - 100; px < qx + 10; px++) {
                        if (px < 0 || px >= width || py < 0 || py >= height) {

                        }
                        else {
                            int e = Difference(keyframe, temporal, px, py, qx, qy);
                            e += lambda_col * Difference(col_origin, col_target, px, py, qx, qy) + lambda_pos * Difference(pos_origin, pos_target, px, py, qx, qy) + lambda_edge * Difference(edge_origin, edge_target, px, py, qx, qy);

                            if (e < minE) {
                                minE = e;
                                temppy = py;
                                temppx = px;
                            }
                        }

                    }
                }
                

                key_origin.pixelColor(qx, qy, key.pixelColor(temppx, temppy));
            }
        }
        std::cout << "Frame " + to_string(i)+" is finished." << endl;
        key_origin.write(".\\result\\" + fixedLength(i, 3) + ".jpg");
        //std::cout << i << endl;

    }
    return 0;

}

int Difference(Mat f1, Mat f2, int px, int py, int qx, int qy) {
    int result = 0;
    int r = abs(f1.at<cv::Vec3b>(py, px)[0] - f2.at<cv::Vec3b>(qy, qx)[0]);
    int g = abs(f1.at<cv::Vec3b>(py, px)[1] - f2.at<cv::Vec3b>(qy, qx)[1]);
    int b = abs(f1.at<cv::Vec3b>(py, px)[2] - f2.at<cv::Vec3b>(qy, qx)[2]);

    result = r + g + b;
    return result * result;

}

std::string fixedLength(int value, int digits = 3) {
    unsigned int uvalue = value;
    std::string result;
    while (digits-- > 0) {
        result += ('0' + uvalue % 10);
        uvalue /= 10;
    }
    std::reverse(result.begin(), result.end());
    return result;
}


void EdgeGuide(string source, string distination) {
    Image target;
    target.read(source);

    //getting edge guide
    Image color_guide(target);
    Image blur(target);
    Image edge_guide(target);
    //gaussian blur and image subtraction as per paper
    blur.gaussianBlur(4, 5);
    edge_guide.composite(blur, 0, 0, MinusDstCompositeOp);
    //highlight edges
    edge_guide.edge();
    //grayscale image
    edge_guide.quantizeColorSpace(GRAYColorspace);
    edge_guide.quantizeColors(256);
    edge_guide.quantize();
    //negate color of image
    edge_guide.negate();
    edge_guide.write(distination);

}

void TemporalGuide(int f, string targetFolder) {
    Mat first, first_gray, second, second_gray, flow_n, pre, result, base;
    base = cv::imread(".\\result\\" + fixedLength(f - 1, 3) + ".jpg", 1);
    first = cv::imread(targetFolder + fixedLength(f - 1, 3) + ".jpg", 1);
    second = cv::imread(targetFolder + fixedLength(f, 3) + ".jpg", 1);
    cvtColor(first, first_gray, CV_BGR2GRAY);
    cvtColor(second, second_gray, CV_RGB2GRAY);
    calcOpticalFlowFarneback(first_gray, second_gray, flow_n, 0.5, 3, 15, 3, 5, 1.2, 0);
    createNewFrame(result, flow_n, 1, base);

    Mat mask, rest;
    inRange(result, Scalar(0, 0, 0), Scalar(5, 5, 5), mask);
    bitwise_or(result, base, rest, mask);
    result = result + rest;
    imwrite(".\\temporal\\" + fixedLength(f) + ".jpg", result);
}

int Difference(String f1, String f2, int px, int py, int qx, int qy) {
    Mat first, second;
    first = cv::imread(f1);
    second = cv::imread(f2);

    int result = 0;

    int r = abs(first.at<cv::Vec3b>(py, px)[0] - second.at<cv::Vec3b>(qy, qx)[0]);
    int g = abs(first.at<cv::Vec3b>(py, px)[1] - second.at<cv::Vec3b>(qy, qx)[1]);
    int b = abs(first.at<cv::Vec3b>(py, px)[2] - second.at<cv::Vec3b>(qy, qx)[2]);

    result += r + g + b;

    return result * result;
}

int PositionalGuide(int f1, int f2, string targetFolder) {
    Mat first, first_gray, second, second_gray, flow_n, base;
    base = cv::imread("gradient.jpg", 1);
    int count = f1;
    Mat result;
    do {
        first = cv::imread(targetFolder + fixedLength(count) + ".jpg", 1);
        second = cv::imread(targetFolder + fixedLength(count + 1) + ".jpg", 1);

        cvtColor(first, first_gray, CV_BGR2GRAY);
        cvtColor(second, second_gray, CV_RGB2GRAY);
        calcOpticalFlowFarneback(first_gray, second_gray, flow_n, 0.5, 3, 15, 3, 5, 1.2, 0);
        createNewFrame(result, flow_n, 1, base);
        base = result;
        count++;
    } while (count < f2);
    Mat mask, rest;
    inRange(result, Scalar(0, 0, 0), Scalar(5, 5, 5), mask);
    base = cv::imread("gradient.jpg", 1);
    bitwise_or(result, base, rest, mask);
    result = result + rest;
    imwrite(".\\positional\\" + fixedLength(f1) + "_" + fixedLength(f2) + ".jpg", result);
    return 0;
}



void createNewFrame(cv::Mat& frame, const cv::Mat& flow, float shift, cv::Mat& next) {
    cv::Mat mapX(flow.size(), CV_32FC1);
    cv::Mat mapY(flow.size(), CV_32FC1);
    cv::Mat newFrame;
    for (int y = 0; y < mapX.rows; y++) {
        for (int x = 0; x < mapX.cols; x++) {
            cv::Point2f f = flow.at<cv::Point2f>(y, x);
            mapX.at<float>(y, x) = x + f.x * shift;
            mapY.at<float>(y, x) = y + f.y * shift;
        }
    }
    remap(next, newFrame, mapX, mapY, cv::INTER_LANCZOS4);
    frame = newFrame;
}

int Gradient(int width, int height) {
    Image gradient(Geometry(width, height), "blue");
    for (int w = 0; w < width; w++) {
        for (int h = 0; h < height; h++) {
            double w_color = w / (double)width;
            double h_color = h / (double)height;
            gradient.pixelColor(w, h, ColorRGB(w_color, h_color, 0));
        }
    }
    gradient.write("gradient.jpg");
    return 0;
}
