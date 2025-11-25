#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int mp_slider = 0;
int th_slider = 0;
int ad_slider = 0;
int block_size_slider = 51;

int th_otsu_value = 0;
int threshold_value = 127;
int threshold_type = 0;
int const max_value = 255;
int const max_type = 4;
int const max_binary_value = 255;

Mat src, src_gray, frame_thresholded;
const char *window_name = "Threshold";

const char *morphology_type = "0 Off\n1 On";
const char *th_method_type = "0M 1O 2A";
const char *th_adaptive_type = "0Men 1Gaus";
const char *blocksize_type = "3 - 105";
const char *trackbar_type = "B I T Z ZI";
const char *trackbar_value = "Value";

static void Threshold(int, void *)
{
    switch (th_slider)
    {
    case 0:
        threshold(src_gray, frame_thresholded, threshold_value, max_binary_value, threshold_type);
        break;
    case 1:
        th_otsu_value = threshold(src_gray, frame_thresholded, threshold_value, max_binary_value, threshold_type + THRESH_OTSU);
        cout << th_otsu_value << endl;
        break;
    case 2:
        adaptiveThreshold(src_gray, frame_thresholded, max_binary_value, ad_slider, threshold_type < 2 ? threshold_type : 1, block_size_slider, 0);
        break;
    }
}

void on_bs_trackbar(int, void *)
{
    block_size_slider = block_size_slider < 3 ? 3 : block_size_slider;

    if (!(block_size_slider % 2))
    {
        block_size_slider++;
    }
    cout << block_size_slider << endl;

    Threshold(0, 0);
}

int main(int argc, char **argv)
{
    VideoCapture cap(0, cv::CAP_V4L2);

    if (!cap.isOpened())
    {
        cout << "Capture could not be opened succesfully" << endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    namedWindow("Video", WINDOW_AUTOSIZE);
    namedWindow(window_name, WINDOW_NORMAL);

    createTrackbar(morphology_type, window_name, &mp_slider, 1, Threshold);
    createTrackbar(th_method_type, window_name, &th_slider, 2, Threshold);
    createTrackbar(th_adaptive_type, window_name, &ad_slider, 1, Threshold);
    createTrackbar(blocksize_type, window_name, &block_size_slider, 105, on_bs_trackbar);
    createTrackbar(trackbar_type, window_name, &threshold_type, max_type, Threshold);
    createTrackbar(trackbar_value, window_name, &threshold_value, max_value, Threshold);

    while (char(waitKey(1)) != 'q' && cap.isOpened())
    {
        Mat frame;
        cap >> frame;

        flip(frame, frame, 1);

        cvtColor(frame, src_gray, COLOR_BGR2GRAY);
        Threshold(0, 0);

        if (mp_slider)
        {
            Mat str_el = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
            morphologyEx(frame_thresholded, frame_thresholded, MORPH_OPEN, str_el);
            morphologyEx(frame_thresholded, frame_thresholded, MORPH_CLOSE, str_el);
        }

        imshow("Video", frame);
        imshow(window_name, frame_thresholded);
    }

    return 0;
}