#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    Mat im = imread("../img/lena.png");

    if (im.empty())
    {
        cout << "Erro!" << endl;
        return -1;
    }

    imshow("Imagem C++", im);
    waitKey(0);
}
