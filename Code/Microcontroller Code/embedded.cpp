/*
*
* Team Id: 
* Author List: Harinandan Teja, Sundeep, Sumanth, Dinesh
*
* Filename: embedded.cpp
* Theme:
<Theme name -- Specific to eYRC>
* Functions: help(), angle( Point, Point, Point), findSquares( const Mat&, vector<vector<Point> >& ),
             findCentroid(vector<Point>), detectFinger(Mat,int), 
             drawSquares( Mat&, const vector<vector<Point> >& ,int),
             orderVertices(vector<Point>), main(int, char**)
* Global Variables: width, height, click_count, thresh, N, wndname_1, wndname_2, frame1, frame2
*
*/
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

/*
*frame1: Saved detected rectangle frame from camera1
*frame2: Saved detected rectangle frame from camera1
*/
vector<Point> frame1,frame2;

//width: Width of the perspective frame
double width = 638;
//height: Width of the perspective frame
double height = 358;
//click_count: Number of clicks detected
int click_count = 0;
/*
*thresh: threshold used while finding contours
*N: Number of threshold levels
*/
int thresh = 50, N = 11;
//wndname_1: Window name for camera 1
const char* wndname_1 = "Camera 1";
//wndname_2: Window name for camera 2
const char* wndname_2 = "Camera 2";


/*
*
* Function Name: print_info
* Input: None
* Output: Void
* Logic: To print project information
*
* Example Call: print_info()
*
*/
static void print_info()
{
    cout <<
    "\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
    "memory storage (it's got it all folks) to find\n"
    "squares in a list of images pic1-6.png\n"
    "Returns sequence of squares detected on the image.\n"
    "the sequence is stored in the specified memory storage\n"
    "Call:\n"
    "./squares\n"
    "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}

/*
*
* Function Name: angle
* Input: pt1 -> point 1
         pt2 -> point 2
         pt0 -> point 0
* Output: gives angle between vectors pt0->pt1 & pt0->pt2
* Logic: Typical mathematical equation
*
* Example Call: angle(Point(1,0),Point(0,1),Point(0,0))
*
*/
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

/*
*
* Function Name: findSquares
* Input: image -> source image for which squares are to be detected
         squares -> output(sequence of squares) is stored in this 
* Output: Void
* Logic: First we find the contours in the image , then we approximate them to polygons , then we find the polygons 
         with 4 vertices and store them in sqaures
*
* Example Call: findSquares(src, squares)
*
*/
static void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    /*
    *gray0: Grayscale image of image
    *gray: image after applying canny
    */
    Mat gray0(image.size(), CV_8U), gray;

    // Convert rgb image to grayscale image
    cv::cvtColor(image, gray0, CV_BGR2GRAY);
    vector<vector<Point> > contours;

    // Find squares in every color plane of the image
    for( int c = 0; c < 1; c++ )
    {
        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                //dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            //approx: approximate contour with accuraccy proportional to the contour perimeter
            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    //maxCosine: maximum cosine of the angle between joint edges
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                        squares.push_back(approx);
                }
            }
        }
    }
}

/*
*
* Function Name: findCentroid
* Input: contour -> set of points for which centroid needs to calculated
* Output: gives centroid of contour(point)
* Logic: Typical mathematical equation (mean of all points)
*
* Example Call: findCentroid(contour)
*
*/
Point findCentroid(vector<Point> contour)
{
    //cent: Centroid of the contour
    Point cent(0,0);
    for(int i=0;i<contour.size();i++){
        cent.x+=contour[i].x;
        cent.y+=contour[i].y;
    }
    cent.x/=contour.size();
    cent.y/=contour.size();
    return cent;
}

/*
*
* Function Name: detectFinger
* Input: src -> source image where finger position is to be detected
         window_tag => tag determines which camera image
* Output: gives finger tip position (point)
* Logic: src image is a binary image. First we find contours of this image then we remove those having contour 
         area less than a threshold value. Then we find the contour with maximum area and find the centroid of
         contour. 
*
* Example Call: detectFinger(src,1)
*
*/
Point detectFinger(Mat src,int window_tag){
    //p: Finger position
    Point p;
    //contours: Contours of the src image
    vector<vector<Point> > contours;
    //hierarchy: hierarchy of contours
    vector<Vec4i> hierarchy;
    //dst: Destination image
    Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
    findContours( src, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    if(contours.size()==0) return Point(-1,-1);
    
    //index: index of the contour with maximum area
    int index = -1;
    //max_area: maximum area among all contours
    double max_area = 0;

    // find the contour with maximum area
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0] )
    {
        if(max_area< contourArea(contours[idx])){
            index = idx;
            max_area = contourArea(contours[idx]);
        }
    }

    // return if no contour is found or the maximum contours area is less than 20(threshold)
    if(index == -1 || max_area<20) return Point(-1,-1);

    //color: random generated color to color the contour
    Scalar color( rand()&255, rand()&255, rand()&255 );
    // fills the contour with the random color
    drawContours( dst, contours, index, color, CV_FILLED, 8, hierarchy );

    // depending on the tag image is show in its respective window
    if(window_tag == 1){
        namedWindow( "Components camera1", 1 );
        imshow( "Components camera1", dst );
    }
    else{
        namedWindow( "Components camera2", 1 );
        imshow( "Components camera2", dst );   
    }
    return findCentroid(contours[index]);
}

/*
*
* Function Name: drawSquares
* Input: image -> source image on which squares are drawn
         squares -> sequence of squares found using detectSquares function
         window_tag -> tag determines which camera image
* Output: Void
* Logic: Squares contain all squares found in the image. First we remove the square with maximum area which is the bounding 
         square of the entire image. Next we find the square with maximum area and use polyline function to draw 
*
* Example Call: drawSquares(src,1)
*
*/
static void drawSquares( Mat& image, const vector<vector<Point> >& squares,int camera_tag)
{
    // return if there are no squares
    if(squares.size() == 0) return;

    if(squares.size() == 1) {
        //p: starting point of the square
        const Point* p = &squares[0][0];
        int n = (int)squares[0].size();
        polylines(image, &p, &n, 1, true, Scalar(255,0,0), 3, LINE_AA);

        // fix the frame for that particular camera
        if(camera_tag == 1) {
            frame1 = squares[0];
        }
        else{
            frame2 = squares[0];
        }
        return;
    }
    //filtered: contains only squares with area difference less than 40000(threshold)
    vector<vector<Point> > filtered;
    //max_area: maximum area among the contours
    double max_area = 0;
    //index: index of the maximum contour
    int index = 0;
    
    // find the maximum area contour
    for(int i=0;i<squares.size();i++){
        if(max_area < contourArea(squares[i])) {
            max_area = contourArea(squares[i]);
        }
    }
    // filter those with area difference less than 40000
    for(int i=0;i<squares.size();i++){
        if(max_area - contourArea(squares[i]) > 40000){
            filtered.push_back(squares[i]);
        }
    }

    // find the maximum again among the filtered squares
    for(int i=0;i<filtered.size();i++){
        if(max_area < contourArea(filtered[i])) {
            max_area = contourArea(filtered[i]);
            index = i;
        }
    }
    if(filtered.size()==0) return;

    //p: starting point of the maximum contour
    const Point* p = &filtered[index][0];
    int n = (int)filtered[index].size();
    polylines(image, &p, &n, 1, true, Scalar(255,0,0), 3, LINE_AA);

    // fix the frame for that particular camera and show in its respective window
    if(camera_tag == 1) {
        frame1 = filtered[index];
        imshow(wndname_1, image);    
    }
    else{
        frame2 = filtered[index];
        imshow(wndname_2, image);
    }
}

/*
*
* Function Name: orderVertices
* Input: frame -> 4 vertices of the rectangle
* Output: set of vertices in correct order (clock wise)
* Logic: find the vertices with min x+y and maximum x+y these are the start and its opposite vertices. Then find the vertex
         with maximum x value this will be the second vertex and one with minimum x will be the last vertex. 
*
* Example Call: orderVertices(frame)
*
*/
static vector<Point> orderVertices(vector<Point> frame)
{
    /*
    *min_point: starting point(minimum x+y)
    *max_point: 3rd point(maximum x+y)
    */
    Point min_point,max_point;
    /*
    *min_sum: minimum sum(x+y)
    *max_sum: maximum sum(x+y)
    */
    double min_sum=frame[0].x+frame[0].y,max_sum = 0;
    //indexi: index of ith point in order
    int index1,index2 = 0,index3,index4;
    // find maximum and minimum sum point
    for(int i=0;i<frame.size();i++){
        if(max_sum < frame[i].x+frame[i].y){
            max_sum = frame[i].x+frame[i].y;
            index1 = i;
        }
        if(min_sum > frame[i].x+frame[i].y){
            min_sum = frame[i].x+frame[i].y;
            index2 = i;
        }
    }
    for(int i=0;i<4;i++){
        if(i!=index1&&i!=index2) {
            index3 = i;
            break;
        }
    }
    for(int i=0;i<4;i++){
        if(i!=index1&&i!=index2&&i!=index3) {
            index4 = i;
            break;
        }
    }

    //output: vector of points in order
    vector<Point> output;
    output.push_back(frame[index2]);
    if(frame[index3].x > frame[index4].x) {
        output.push_back(frame[index3]);
        output.push_back(frame[index1]);
        output.push_back(frame[index4]);
    }
    else {
        output.push_back(frame[index4]);
        output.push_back(frame[index1]);
        output.push_back(frame[index3]);
    }
    return output;
}

/*
* Function Name: main
* Input: None
* Output: int to inform the caller that the program exited correctly or
* incorrectly (C code standard)
* Logic: Initialize the cameras , get images and apply the algorithm to find finger positions and touch
* Example Call: Called automatically by the Operating System
*
*/
int main(int /*argc*/, char** /*argv*/)
{
    // print starup info
    print_info();
    // create two windows for two cameras
    namedWindow( wndname_1, 1 );
    namedWindow( wndname_2, 1 );
    //squares_i: squares detected in captured image from camera i
    vector<vector<Point> > squares_1,squares_2;
    
    //cap_i: image capture of camera i
    VideoCapture cap_1 = VideoCapture(1);
    VideoCapture cap_2 = VideoCapture(2);
    cout<<"Press s to save the frame for first camera"<<endl;
    while(true){
        frame1.clear();
        //image: captured image from camera 1
        Mat image;
        cap_1 >> image;

        if(image.empty()){
            cout << "Couldn't load.." << endl;
            continue;
        }
        // find and draw squares in image
        findSquares(image, squares_1);
        drawSquares(image, squares_1,1);

        if(frame1.size() == 0) {
            imshow(wndname_1,image);
            if(waitKey(30) == 's') {
                cout<<"Frame saved........"<<endl;
                break;
            }
            continue;
        }
        // order the vertices
        frame1 = orderVertices(frame1);
        //box: bounding box for perspective tranformation
        RotatedRect box = minAreaRect(Mat(frame1));
        //inputQuad: input quadrilateral for perspective tranformation
        Point2f inputQuad[4]; 
        //outputQuad: output quadrilateral for perspective tranformation
        Point2f outputQuad[4];

        inputQuad[0] = frame1[0];
        inputQuad[1] = frame1[1];
        inputQuad[2] = frame1[2];
        inputQuad[3] = frame1[3];

        outputQuad[0] = Point(0,0);
        outputQuad[3] = Point(0,height);
        outputQuad[2] = Point(width,height);
        outputQuad[1] = Point(width,0);
        //output: output after applying perspective transformation
        Mat output, lambda = getPerspectiveTransform( inputQuad, outputQuad );
        warpPerspective(image,output,lambda,output.size() );
        imshow("Transformed",output);
        
        if(waitKey(30) == 's') {
            cout<<"Frame saved.."<<endl;
            break;
        };
    }
    cout<<"Press s to save the frame for second camera"<<endl;
    while(true){
        frame2.clear();
        //image: captured image from camera 2
        Mat image;
        cap_2 >> image;

        if(image.empty()){
            cout << "Couldn't load.." << endl;
            continue;
        }
        // find and draw squares in image
        findSquares(image, squares_2);
        drawSquares(image, squares_2,2);

        if(frame2.size() == 0) {
            imshow(wndname_2,image);
            if(waitKey(30) == 's') {
                cout<<"Frame saved........"<<endl;
                break;
            }
            continue;
        }
        // order the vertices
        frame2 = orderVertices(frame2);
        //box: bounding box for perspective tranformation
        RotatedRect box = minAreaRect(Mat(frame2));
        //inputQuad: input quadrilateral for perspective tranformation
        Point2f inputQuad[4]; 
        //outputQuad: output quadrilateral for perspective tranformation
        Point2f outputQuad[4];

        inputQuad[0] = frame2[0];
        inputQuad[1] = frame2[1];
        inputQuad[2] = frame2[2];
        inputQuad[3] = frame2[3];

        outputQuad[0] = Point(0,0);
        outputQuad[3] = Point(0,height);
        outputQuad[2] = Point(width,height);
        outputQuad[1] = Point(width,0);
        //output: output after applying perspective transformation
        Mat output, lambda = getPerspectiveTransform( inputQuad, outputQuad );
        warpPerspective(image,output,lambda,output.size() );
        imshow("Transformed",output);
        
        if(waitKey(30) == 's') {
            cout<<"Frame saved.."<<endl;
            break;
        };
    }

    // loop to detect finger position
    while(true){
        // detect finger position in camera 2 perspective
        //image: captured images from the two cameras
        Mat image;
        cap_1 >> image;
        imshow(wndname_1,image);
        /*
        *output: output image after applying perspective transformation
        *hsv_image: image after COLOR_BGR2HSV transformation
        *lower_red_hue_range: image after applying inrange (detects color in the give range)
        *pyr: downscaled image
        *timg: upscaled image
        */
        Mat output,hsv_image,lower_red_hue_range,pyr,timg,lambda;
        //inputQuad: input quadrilateral for perspective tranformation
        Point2f inputQuad[4]; 
        //outputQuad: output quadrilateral for perspective tranformation
        Point2f outputQuad[4];
        outputQuad[0] = Point(0,0);
        outputQuad[3] = Point(0,height);
        outputQuad[2] = Point(width,height);
        outputQuad[1] = Point(width,0);

        inputQuad[0] = frame1[0];
        inputQuad[1] = frame1[1];
        inputQuad[2] = frame1[2];
        inputQuad[3] = frame1[3];
        
        lambda = getPerspectiveTransform( inputQuad, outputQuad );
        warpPerspective(image,output,lambda,output.size() );
        
        // convert from rgb to hsv
        cvtColor(output, hsv_image, cv::COLOR_BGR2HSV);
        // to detect color in the given range (red)
        inRange(hsv_image, Scalar(0,108,122), Scalar(10, 255, 255), lower_red_hue_range);

        // morphological closing (fill small holes in the foreground)
        erode(lower_red_hue_range, lower_red_hue_range, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        // morphological opening (expand small holes in the foreground)
        dilate( lower_red_hue_range, lower_red_hue_range, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
        dilate( lower_red_hue_range, lower_red_hue_range, getStructuringElement(MORPH_ELLIPSE, Size(30, 30)) ); 

        // downscale and upscale the image
        pyrDown(lower_red_hue_range, pyr, Size(lower_red_hue_range.cols/2, lower_red_hue_range.rows/2));
        pyrUp(pyr, timg, lower_red_hue_range.size());
        threshold(timg, timg, 150, 255, CV_THRESH_BINARY | THRESH_OTSU);

        //p_1: finger position wrt camera 1
        Point p_1 = detectFinger(timg,1);
        
        // detect finger position in camera 2 perspective
        cap_2 >> image;
        imshow(wndname_2,image);
        inputQuad[0] = frame2[0];
        inputQuad[1] = frame2[1];
        inputQuad[2] = frame2[2];
        inputQuad[3] = frame2[3];
        
        lambda = getPerspectiveTransform( inputQuad, outputQuad );
        warpPerspective(image,output,lambda,output.size() );
        // convert from rgb to hsv
        cvtColor(output, hsv_image, cv::COLOR_BGR2HSV);
        // to detect color in the given range (red)
        inRange(hsv_image, Scalar(0,108,122), Scalar(10, 255, 255), lower_red_hue_range);

        // morphological closing (fill small holes in the foreground)
        erode(lower_red_hue_range, lower_red_hue_range, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        // morphological opening (expand small holes in the foreground)
        dilate( lower_red_hue_range, lower_red_hue_range, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
        dilate( lower_red_hue_range, lower_red_hue_range, getStructuringElement(MORPH_ELLIPSE, Size(20, 20)) ); 

        // downscale and upscale the image
        pyrDown(lower_red_hue_range, pyr, Size(lower_red_hue_range.cols/2, lower_red_hue_range.rows/2));
        pyrUp(pyr, timg, lower_red_hue_range.size());
        threshold(timg, timg, 150, 255, CV_THRESH_BINARY | THRESH_OTSU);

        //p_2: finger position wrt camera 2
        Point p_2 = detectFinger(timg,2);

        // find the distance between the points 
        if(p_2.x!=-1 && p_2.y!=-1  && p_1.x!=-1 && p_1.y!=-1) {
            double distance = norm(Mat(p_1),Mat(p_2));            
            //mid: mid point of the two finger positions from two cameras
            Point mid;
            mid.x = (p_1.x+p_2.x)*1920/(2*width);
            mid.y = (p_1.y+p_2.y)*1080/(2*height);
            // click detected
            if(distance<50) {
                string script = "./mousecontrol.sh " + to_string(mid.x) +" "+ to_string(mid.y);
                const char *cstr = script.c_str();
                system(cstr);
                click_count++;
                if(click_count==3) {
                    /*script = "./click.sh";
                    const char *cstr1 = script.c_str();
                    system(cstr1);  */
                    click_count = 0;
                }
            }
            else if(distance<150){
                string script = "./mousecontrol.sh " + to_string(mid.x) +" "+ to_string(mid.y);
                const char *cstr = script.c_str();
                system(cstr);
                click_count = 0;
            }

        }
        if(waitKey(30) == 'x') {
            cout<<"Exiting.."<<endl;
            break;
        };
    }

    return 0;
}
