#include <Magick++.h> 
#include <iostream> 
using namespace std;
using namespace Magick;
int main(int argc, char** argv)
{
    InitializeMagick(*argv);

    // Construct the image object. Seperating image construction from the 
    // the read operation ensures that a failure to read the image file 
    // doesn't render the image object useless. 
    Image target;

    try {
        // Read a file into image object
        target.read(".\\resources\\000_target.jpg");
        
        //getting edge guide
        Image color_guide(target);
        Image blur(target);
        Image edge_guide(target);
        blur.gaussianBlur(4, 5);
        edge_guide.composite(blur, 0, 0, MinusDstCompositeOp);
        edge_guide.quantizeColorSpace(GRAYColorspace);
        edge_guide.quantizeColors(256);
        edge_guide.quantize();
        edge_guide.negate();
        edge_guide.write("edge_guide.jpg");

        //get optical flow
        Image target2(".\\resources\\010_t.jpg");
        Image optical_flow(target2);
        //optical_flow.composite(target,)

   



       
    }
    catch (Exception& error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    return 0;
}


