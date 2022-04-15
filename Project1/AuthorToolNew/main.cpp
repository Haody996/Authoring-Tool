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

std::string fixedLength(int value, int digits);
int Gradient(int, int);
int PositionalGuide(int, int);

void EdgeGuide(string source, string distination);
void createNewFrame(cv::Mat& frame, const cv::Mat& flow, float shift, cv::Mat& next);


int main(int argc, char** argv)
{
    InitializeMagick(*argv);

    // Construct the image object
    Image target;

    //for optical flow
    //string video = ".\\resources\\test.mov";

    //base for positional guide
    target.read(".\\resources\\target\\000.jpg");
    unsigned int w = target.size().width();
    unsigned int h = target.size().height();
    Gradient(w, h);

    //get any positional guide, e.g. between keyframe 0 and target frame 10
    PositionalGuide(0,10);
    try{
        EdgeGuide(".\\resources\\000_key.jpg", ".\\keyframe_guides\\edge.jpg");
        for (int i = 0; i < 100; i++) {
            string s = fixedLength(i, 3);
            //target.read(".\\resources\\target\\000.jpg");
            EdgeGuide(".\\resources\\target\\" + s + ".jpg", ".\\edges\\" + s + ".jpg");

        }
       
    }
    catch (Magick::Exception & error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    
    return 0;

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

void TemporalGuide(int f) {
    Mat first, first_gray, second, second_gray, flow_n, pre, result;
    pre = cv::imread(".\\result\\" + fixedLength(f-1) + ".jpg", 1);
    first = cv::imread(".\\resources\\target\\" + fixedLength(f-1) + ".jpg", 1);
    second = cv::imread(".\\resources\\target\\" + fixedLength(f) + ".jpg", 1);
    cvtColor(first, first_gray, CV_BGR2GRAY);
    cvtColor(second, second_gray, CV_RGB2GRAY);
    calcOpticalFlowFarneback(first_gray, second_gray, flow_n, 0.5, 3, 15, 3, 5, 1.2, 0);
    createNewFrame(result, flow_n, 1, pre);
    imwrite(".\\temporal\\" + fixedLength(f) + ".jpg", result);
}

int PositionalGuide(int f1,int f2) {
    Mat first, first_gray, second, second_gray, flow_n, base;
    base = cv::imread("gradient.jpg", 1);
    int count = f1;
    Mat remap_n;
    do {
        first = cv::imread(".\\resources\\target\\" + fixedLength(count) + ".jpg", 1);
        second = cv::imread(".\\resources\\target\\" + fixedLength(count + 1) + ".jpg", 1);
        
        cvtColor(first, first_gray, CV_BGR2GRAY);
        cvtColor(second, second_gray, CV_RGB2GRAY);
        calcOpticalFlowFarneback(first_gray, second_gray, flow_n, 0.5, 3, 15, 3, 5, 1.2, 0);
        createNewFrame(remap_n, flow_n, 1, base);
        base = remap_n;
        count++;
        //imwrite(".\\positional\\" + fixedLength(f1) + "_" + fixedLength(f2) + "_" + to_string(count) + ".jpg", remap_n);
    } while (count < f2);

    //imshow("remap_n", remap_n);
    //imwrite(".\\positional\\" + fixedLength(f1) + "_" + fixedLength(f2) + ".jpg", remap_n);
    Mat mask, rest;
    inRange(remap_n, Scalar(0,0,0), Scalar(5,5,5), mask);
    base = cv::imread("gradient.jpg", 1);
    bitwise_or(remap_n, base, rest,mask);
    remap_n = remap_n + rest;
    imwrite(".\\positional\\"+ fixedLength(f1) + "_" +fixedLength(f2) + ".jpg", remap_n);
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
    Image gradient(Geometry(width,height), "blue");
    for (int w = 0; w < width; w++) {
        for (int h = 0; h < height; h++) {
            double w_color = w / (double) width;
            double h_color = h / (double) height;
            gradient.pixelColor(w, h, ColorRGB(w_color , h_color, 0));
        }
    }
    gradient.write("gradient.jpg");
    return 0;
}