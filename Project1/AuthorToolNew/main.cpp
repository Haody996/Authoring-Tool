#include <Magick++.h> 
#include <iostream> 
#include <iomanip>
using namespace std;
using namespace Magick;

std::string fixedLength(int value, int digits);

int main(int argc, char** argv)
{
    InitializeMagick(*argv);

    // Construct the image object. Seperating image construction from the 
    // the read operation ensures that a failure to read the image file 
    // doesn't render the image object useless. 
    Image target;
    //for optical flow
    string video = ".\\resources\\test.mov";
    try {

        for (int i = 0; i < 100; i++) {
            string s = fixedLength(i, 3);

            target.read(".\\resources\\target\\" + s + ".jpg");

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
            edge_guide.write(".\\edges\\"+s+".jpg");

            //get optical flow

        }
       
    }
    catch (Exception& error_)
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
