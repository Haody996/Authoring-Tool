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
int Difference(String f1, String f2, int px, int py, int qx, int qy);
void TemporalGuide(int f);
int Difference(Mat f1, Mat f2, int px, int py, int qx, int qy);

String targetFolder = ".\\resources\\target\\";
String keyframeFolder = ".\\keyframe_guides\\";
String keyframeLocation = ".\\keyframe_guides\\keyframe.jpg";

int main(int argc, char** argv)
{
    InitializeMagick(*argv);

    // Construct the image object
    Image target;

    Mat target_000;
    target_000 = imread(targetFolder+"000.jpg");
    int width = target_000.size().width;
    int height = target_000.size().height;

    

    //weights for error function
    double lambda_col = 7;
    double lambda_pos = 1;
    double lambda_edge = 2;
    double lambda_temp = 0.5;

    
    //base for positional guide
    target.read(targetFolder+"000.jpg");
    unsigned int w = target.size().width();
    unsigned int h = target.size().height();
    Gradient(w, h);

    //get any positional guide, e.g. between keyframe 0 and target frame 10
    /*
    try{
        //EdgeGuide(".\\resources\\000_key.jpg", ".\\keyframe_guides\\edge.jpg");
        for (int i = 0; i < 100; i++) {
            string s = fixedLength(i, 3);
            //target.read(".\\resources\\target\\000.jpg");
            EdgeGuide(targetFolder + s + ".jpg", ".\\edges\\" + s + ".jpg");

        }
       
    }
    catch (Magick::Exception & error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    
    
    for (int i = 1; i < 100; i++) {
        PositionalGuide(0, i);
    }
    */


    Image key,key_origin;
    key.read(keyframeLocation);
    key_origin.read(targetFolder + "000.jpg");

    Mat col_origin = imread(keyframeFolder + "origin.jpg");
    Mat pos_origin = imread(keyframeFolder + "gradient.jpg");
    Mat edge_origin = imread(keyframeFolder + "edge.jpg");
    Mat keyframe = imread(keyframeLocation);


    for (int i = 1; i < 100; i++) {
        Mat prev = imread(".\\result_test\\" + fixedLength(i - 1, 3) + ".jpg");
        //Mat prev_blurred;
        //GaussianBlur(prev, prev_blurred,Size(5,5),1);
        //imwrite(".\\blur\\" + fixedLength(i, 3) + ".jpg", prev_blurred);
        
        
        TemporalGuide(i);
        Mat temporal = imread(".\\temporal\\" + fixedLength(i,3) + ".jpg");


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
         
                for (int py = qy-20; py < qy+20; py++) {
                    for (int px = qx-150; px < qx+30; px++) {
                        if (px < 0 || px >= width || py<0 || py >= height) {

                        }
                        else {
                            int e = Difference(keyframe,prev, px, py, qx, qy);

                            //e += lambda_col * Difference(col_origin, col_target, px, py, qx, qy) + lambda_pos * Difference(pos_origin,pos_target,px,py,qx,qy) + lambda_edge * Difference(edge_origin,edge_target,px,py,qx,qy);
                            e += lambda_col * Difference(col_origin, col_target, px, py, qx, qy) + lambda_pos * Difference(pos_origin, pos_target, px, py, qx, qy) + lambda_edge * Difference(edge_origin, edge_target, px, py, qx, qy);

                            if (e < minE) {
                                minE = e;
                                temppy = py;
                                temppx = px;
                            }
                        }
                    
                    }
                }
                //cout << qx << endl;
                //cout << qy << endl;
 
                key_origin.pixelColor(qx, qy, key.pixelColor(temppx, temppy));
            }
        }
        key_origin.write(".\\result_test\\"+fixedLength(i,3)+".jpg");
        std::cout << i << endl;

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

void TemporalGuide(int f) {
    Mat first, first_gray, second, second_gray, flow_n, pre, result, base;
    base = cv::imread(".\\result_test\\" + fixedLength(f-1,3) + ".jpg", 1);
    first = cv::imread(targetFolder + fixedLength(f-1,3) + ".jpg", 1);
    second = cv::imread(targetFolder + fixedLength(f,3) + ".jpg", 1);
    cvtColor(first, first_gray, CV_BGR2GRAY);
    cvtColor(second, second_gray, CV_RGB2GRAY);
    calcOpticalFlowFarneback(first_gray, second_gray, flow_n, 0.5, 3, 15, 3, 5, 1.2, 0);
    createNewFrame(result, flow_n, 1, pre);

    //Mat mask, rest;
    //inRange(result, Scalar(0, 0, 0), Scalar(5, 5, 5), mask);
    //bitwise_or(result, base, rest, mask);
    //result = result + rest;
    imwrite(".\\temporal\\" + fixedLength(f) + ".jpg", result);
}

int Difference(String f1,String f2, int px, int py, int qx, int qy) {
    Mat first, second;
    first = cv::imread(f1);
    second = cv::imread(f2);

    int result = 0;

    int r = abs(first.at<cv::Vec3b>(py, px)[0] - second.at<cv::Vec3b>(qy, qx)[0]);
    int g = abs(first.at<cv::Vec3b>(py, px)[1] - second.at<cv::Vec3b>(qy, qx)[1]);
    int b = abs(first.at<cv::Vec3b>(py, px)[2] - second.at<cv::Vec3b>(qy, qx)[2]);

    result += r + g + b;

    return result*result;
}

int PositionalGuide(int f1,int f2) {
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
    inRange(result, Scalar(0,0,0), Scalar(5,5,5), mask);
    base = cv::imread("gradient.jpg", 1);
    bitwise_or(result, base, rest,mask);
    result = result + rest;
    imwrite(".\\positional\\"+ fixedLength(f1) + "_" +fixedLength(f2) + ".jpg", result);
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
