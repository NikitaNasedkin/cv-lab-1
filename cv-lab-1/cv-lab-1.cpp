#include <iostream>
#include <opencv2/opencv.hpp>


using namespace cv;

int main()
{
    Mat srcImg = imread("Lenna.png");
    CascadeClassifier faceClassifier;
    faceClassifier.load("haarcascade_frontalface_alt2.xml");
    std::vector<Rect> faces;

    // task 1
    faceClassifier.detectMultiScale(srcImg, faces, 1.1, 2);
    Mat imgWithFaceRec = srcImg.clone();
    rectangle(imgWithFaceRec, faces[0].tl(), faces[0].br(), Scalar(255, 255, 255), 2);

    imshow("Detected Face", imgWithFaceRec);
    waitKey(0);

    // task 2
    double crop = 0.2;
    cv::Size newSize(faces[0].width * crop, faces[0].height * crop);
    cv::Point indent(newSize.width / 2, newSize.height / 2);

    faces[0] = faces[0] + newSize;
    faces[0] = faces[0] - indent;
    rectangle(imgWithFaceRec, faces[0].tl(), faces[0].br(), Scalar(255, 255, 255), 2);

    Mat croppedFace = srcImg(faces[0]);

    imshow("Crop", croppedFace);
    waitKey(0);

    // task 3
    Mat borders = Mat::zeros(croppedFace.size(), CV_8UC3);
    std::vector<std::vector<cv::Point>> circuit;
    Mat canny;
    Canny(croppedFace, canny, 100, 200);

    findContours(canny, circuit, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    drawContours(borders, circuit, -1, cv::Scalar(255, 255, 255), 1);

    imshow("canny", borders);
    waitKey(0);

    // task 4
    Mat newBorders = Mat::zeros(croppedFace.size(), CV_8UC3);
    circuit.erase(std::remove_if(circuit.begin(), circuit.end(), [](std::vector<Point> const& contour) {
            return arcLength(contour, false) <= 10;
        }),
        circuit.end());

    cv::drawContours(newBorders, circuit, -1, Scalar(255, 255, 255), 1);

    imshow("clear canny", newBorders);
    waitKey(0);

    // task 5
    Mat increase;
    Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    dilate(newBorders, increase, kernel);

    imshow("increase", increase);
    waitKey(0);

    // task 6
    Mat gauss, M;
    GaussianBlur(increase, gauss, Size(5, 5), 3);
    normalize(gauss, M, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);

    imshow("gauss", gauss);
    waitKey(0);

    // task 7
    Mat F1;
    bilateralFilter(croppedFace, F1, 15, 80, 80);

    imshow("F1", F1);
    waitKey(0);

    // task 8
    double sigma = 1, amount = 3;
    cv::Mat blurry, sharp, F2;
    cv::GaussianBlur(croppedFace, blurry, cv::Size(), sigma);
    cv::addWeighted(croppedFace, 1 + amount, blurry, -amount, 0, F2);

    imshow("F2", F2);
    waitKey(0);

    // task 9
    Mat result = Mat::zeros(croppedFace.size(), CV_8UC3);
    for (int y = 0; y < croppedFace.rows; y++)
    {
        for (int x = 0; x < croppedFace.cols; x++)
        {
            Vec3b F1_pixel = F1.at<Vec3b>(y, x);
            Vec3b F2_pixel = F2.at<Vec3b>(y, x);
            float M_pixel = M.at<float>(y, x);

            Vec3b res_pixel;
            for (int c = 0; c < 3; c++)
            {
                res_pixel[c] = M_pixel * F2_pixel[c] + (1.0 - M_pixel) * F1_pixel[c];
            }

            result.at<Vec3b>(Point(x, y)) = res_pixel;
        }
    }

    imshow("result", result);
    waitKey(0);

    return 0;
}
