//#include <clm-tracker/myFaceTracker.hpp>
//#include <clm-tracker/FaceTracker.hpp>
#include <tracker/IO.hpp>
#include <tracker/ShapeModel.hpp>
#include <opencv2/opencv.hpp>
#include "shapeRegressor.hpp"
#include <iostream>
#include <sys/time.h>
#define db at<double>
#define it at<int>
using namespace std;
using namespace cv;
using namespace FACETRACKER;
//=========================================================================
void
drawShape(Mat &image,Mat_<float> &shape,Mat &con,
          const Scalar color1 = CV_RGB(255,0,0),
          const Scalar color2 = CV_RGB(255,0,0),
          const vector<int> size = vector<int>())
{
    int i,n = shape.rows/2; Point p1,p2;
    for(i = 0; i < con.cols; i++){
        p1 = Point(shape(2*con.at<int>(0,i)),
                   shape(2*con.at<int>(0,i)+1));
        p2 = Point(shape(2*con.at<int>(1,i)),
                   shape(2*con.at<int>(1,i)+1));
        line(image,p1,p2,color1,1,CV_AA);
    }
    for(i = 0; i < n; i++){
        p1 = Point(shape(2*i),shape(2*i+1));
        if((int)size.size() == n)circle(image,p1,size[i],color2,1,CV_AA);
        else circle(image,p1,2,color2,1,CV_AA);
    }return;
}
//=========================================================================
Mat_<uchar>
norm_illum(const Mat_<uchar> &img,
           const Size size = Size(11,11),
           const float thresh = 0.25)
{
    Mat im; img.convertTo(im,CV_32F);
    Mat A; boxFilter(im,A,CV_32F,size,Point(-1,-1),false);
    A /= size.width*size.height;
    Mat I = im-A,P; P = I.mul(I);
    boxFilter(P,A,CV_32F,size,Point(-1,-1),false);
    A /= size.width*size.height; sqrt(A,A);
    double minVal,maxVal; minMaxLoc(A,&minVal,&maxVal);
    Mat A1; threshold(A,A1,thresh*(minVal+maxVal),1,THRESH_BINARY);
    Mat A2 = 1.0 - A1;
    Mat V1; divide(I,A,V1);
    Mat V2 = I/max(1e-6,(thresh*(minVal+maxVal)));
    I = V1.mul(A1) + V2.mul(A2);
    normalize(I,I,0,255,NORM_MINMAX);
    Mat N; I.convertTo(N,CV_8U); return N;
}
//=========================================================================
void
drawPts(Mat &image,Mat_<float> &shape,
        const Scalar color = CV_RGB(255,0,0))
{
    int i,n = shape.rows/2;
    for(i = 0; i < n; i++)
        circle(image,Point(shape(2*i),shape(2*i+1)),1,color,1,CV_AA);
    return;
}
//=========================================================================
bool
hasNans(const Mat &X)
{
    if(X.type() == CV_32F){
        const float* xp = X.ptr<float>(0); int n = X.rows*X.cols;
        for(int i = 0; i < n; i++,++xp){
            if(cvIsNaN(*xp))return true;
            if(cvIsInf(*xp))return true;
        }return false;
    }else if(X.type() == CV_64F){
        const double* xp = X.ptr<double>(0); int n = X.rows*X.cols;
        for(int i = 0; i < n; i++,++xp){
            if(cvIsNaN(*xp))return true;
            if(cvIsInf(*xp))return true;
        }return false;
    }else return true;
}
//=========================================================================
void
train()
{
    //    Mat con = IO::LoadCon("src/tracker/clm-tracker/resources/face.con");
    //    FaceTracker* tracker =
    //    LoadFaceTracker("src/tracker/clm-tracker/resources/face.mytracker");
    //    PDM3D pdm = ((myFaceTracker*)tracker)->_clm._pdm;
    
    Mat con = IO::LoadCon("tracker/resources/face.con");
    PDM3D pdm("tracker/resources/face.pdm3d");
    
    
    Mat plocal(pdm.nModes(),1,CV_64F),pglobl(6,1,CV_64F);
    /*
     plocal = 0.0; pglobl = (Mat_<double>(6,1) << 3,0,0,CV_PI/40,80,75);
     Mat r(2*pdm.nPoints(),1,CV_64F); int n = pdm.nPoints();
     pdm.CalcShape2D(r,plocal,pglobl); Size size(160,160);
     */
    
    plocal = 0.0; pglobl = (Mat_<double>(6,1) << 1.5,0,0,CV_PI/40,40,37.5);
    
    
    Mat r(2*pdm.nPoints(),1,CV_64F); int n = pdm.nPoints();
    pdm.CalcShape2D(r,plocal,pglobl); Size size(80,80);
    
    Mat_<float> ref(2*n,1);
    for(int i = 0; i < n; i++){ref(2*i) = r.db(i); ref(2*i+1) = r.db(i+n);}
    
    /*
     Mat im = Mat::zeros(size,CV_8UC3);
     drawShape(im,ref,con);
     imshow("test",im); waitKey(0); exit(0);
     */
    //vector<string> imlist = IO::GetList("data2/frontal.imlist");
    //vector<string> lmlist = IO::GetList("data2/frontal.lmlist");
    
    vector<string> imlist = IO::GetList("/Users/Jun/Documents/data/images/66/frontal/imlist");
    vector<string> lmlist = IO::GetList("/Users/Jun/Documents/data/images/66/frontal/lmlist");
    
    int N = imlist.size(); assert((N > 0) && ((int)lmlist.size() == N));
    
    vector<Mat_<float> > landmarks(N);
    for(int k = 0; k < N; k++){
        Mat pt = IO::LoadPts(lmlist[k].c_str()); int n = pt.rows/2;
        landmarks[k].create(2*n,1);
        for(int i = 0; i < n; i++){
            landmarks[k](2*i) = pt.db(i); landmarks[k](2*i+1) = pt.db(i+n);
        }
    }
    
    shapeRegressor R;
    R.train(imlist,landmarks,ref,size,1,200,5,400,1,1000,0.1);
    R.save("shapeRegressor_illum.data");
    
    return;
}
//=========================================================================


// void
// test()
// {
//   shapeRegressor R; R.load("data3/shapeRegressor2.data");
//   Mat con = IO::LoadCon("src/tracker/clm-tracker/resources/face.con");
//   FaceTracker* tracker =
//     LoadFaceTracker("src/tracker/clm-tracker/resources/face.mytracker");
//   PDM3D pdm = ((myFaceTracker*)tracker)->_clm._pdm;
//   Mat plocal(pdm.nModes(),1,CV_64F),pglobl(6,1,CV_64F);
//   Mat s(2*pdm.nPoints(),1,CV_64F); int n = pdm.nPoints();
//   //vector<string> imlist = IO::GetList("data2/frontal.imlist");
//   //vector<string> lmlist = IO::GetList("data2/frontal.lmlist");
//   vector<string> imlist = IO::GetList("/Users/jasonsaragih/Documents/data/Faces/All/cropped/imlist");
//   vector<string> lmlist = IO::GetList("/Users/jasonsaragih/Documents/data/Faces/All/cropped/lmlist");

//   int N = imlist.size(); assert((N > 0) && ((int)lmlist.size() == N));
//   RNG rn(getTickCount());
//   for(int i = 0; i < N; i++){
//     Mat im = imread(imlist[rn.uniform(0,N)],0);
//     Mat pt = IO::LoadPts(lmlist[i].c_str());
//     //rn.fill(s,RNG::UNIFORM,-20,20); pt += s;
//     pdm.CalcParams(pt,plocal,pglobl);
//     plocal = 0.0f;
//     pglobl.db(0) += rn.uniform(-0.1,0.1);
//     pglobl.db(1) = 0;
//     pglobl.db(2) = 0;
//     pglobl.db(3) = 0;
//     pglobl.db(4) += rn.uniform(-5,5);
//     pglobl.db(5) += rn.uniform(-5,5);
//     pdm.CalcShape2D(s,plocal,pglobl); Mat_<float> p(2*n,1);
//     for(int j = 0; j < n; j++){p(2*j) = s.db(j); p(2*j+1) = s.db(j+n);}
//     Mat_<float> p0 = p.clone();
//     for(int iter = 0; iter < R._niter; iter++){
//       Mat img; cvtColor(im,img,CV_GRAY2RGB);
//       Mat imm; resize(img,imm,Size(3*img.cols,3*img.rows));
//       //drawShape(img,p0,con,CV_RGB(255,0,0),CV_RGB(255,0,0));
//       p = R.predict(im,p0,iter);
//       Mat_<float> q = 3.0*p;
//       drawShape(imm,q,con,CV_RGB(0,255,0),CV_RGB(0,255,0));
//       imshow("test",imm); if(waitKey(10) == 27)break;
//     }
//     if(waitKey(0) == 27)break;
//   }return;
// }
//=========================================================================
class faceDet{
public:
    CascadeClassifier _fdet;
    faceDet(){this->init();}
    void init(const char* fname = "model/face.xml"){_fdet.load(fname);}
    vector<Point2f> findEyes(const Mat &img){
        Rect R = this->findFace(img);
        if((R.x < 0) || (R.y < 0))return vector<Point2f>();
        vector<Point2f> eye(2);
        eye[0] = Point2f(R.x+0.314932*R.width,R.y+0.393389*R.height);
        eye[1] = Point2f(R.x+0.683115*R.width,R.y+0.399543*R.height); return eye;
    }
    Rect findFace(const Mat &im){
        Mat img; if(im.channels() == 1)img=im; else cvtColor(im,img,CV_RGB2GRAY);
        vector<Rect> faces;
        Mat smallImg; equalizeHist(img,smallImg);
        _fdet.detectMultiScale(smallImg,faces,1.1, 2, 0
                               |CV_HAAR_FIND_BIGGEST_OBJECT
                               |CV_HAAR_SCALE_IMAGE,Size(30,30));
        if(faces.size() < 1){return Rect(-1,-1,0,0);}
        else return faces[0];
    }
};
//=========================================================================
Mat_<float>
stacked2inter(Mat &x){
    int n = x.rows/2; Mat_<float> y(2*n,1);
    for(int i = 0; i < n; i++){
        y(2*i) = x.db(i); y(2*i+1) = x.db(i+n);
    }return y;
}
//=========================================================================
Mat
inter2stacked(Mat_<float> &x){
    int n = x.rows/2; Mat y(2*n,1,CV_64F);
    for(int i = 0; i < n; i++){
        y.db(i) = x(2*i); y.db(i+n) = x(2*i+1);
    }return y;
}
//=========================================================================
void
test_track()
{
    shapeRegressor R;
    R.load("shapeRegressor2.data");

    cout << R._niter << endl;
    
    //***********
    
    //  for(int iter = 0; iter < R._niter; iter++){
    //    Mat I = Mat::zeros(R._size,CV_8UC3);
    //    Mat con = IO::LoadCon("tracker/resources/face.con");
    //    drawShape(I,R._ref,con,CV_RGB(255,0,0),CV_RGB(255,0,0));
    //    for(int level = 0; level < R._C[iter]._nlevels; level++){
    //      for(int f = 0; f < R._C[iter]._F[level]._nfeats; f++){
    //  int i1 = R._C[iter]._F[level]._index(f,0);
    //	int i2 = R._C[iter]._F[level]._index(f,1);
    //	float x1 = R._C[iter]._F[level]._xloc(f,0) + R._ref(2*i1);
    //	float x2 = R._C[iter]._F[level]._xloc(f,1) + R._ref(2*i2);
    //	float y1 = R._C[iter]._F[level]._yloc(f,0) + R._ref(2*i1+1);
    //	float y2 = R._C[iter]._F[level]._yloc(f,1) + R._ref(2*i2+1);
    //	line(I,Point2f(x1,y1),Point2f(x2,y2),CV_RGB(255,255,255),1,CV_AA);
    //	circle(I,Point2f(x1,y1),1,CV_RGB(0,255,0),1,CV_AA);
    //	circle(I,Point2f(x2,y2),1,CV_RGB(0,255,0),1,CV_AA);
    //      }
    //      imshow("test",I); if(waitKey(0) == 27)break;
    //    }
    //  }
    
    //***********
    
    struct timeval t0, t1;
    
    PDM3D pdm("tracker/resources/face.pdm3d");
    Mat plocal(pdm.nModes(),1,CV_64F),pglobl(6,1,CV_64F);
    int n = pdm.nPoints(); Mat mean = pdm._M.clone();
    Mat con = IO::LoadCon("tracker/resources/face.con");
    VideoCapture cam(0);
    if(cam.isOpened()){
        std::cout << "Camera open" << std::endl;
    }
    
    faceDet fdet; bool init = false;
    Mat s(2*n,1,CV_64F); Mat_<float> r(2*n,1); RNG rn;
    while(1){
        Mat frame; cam >> frame;
        if(frame.empty())continue;
        
        //cv::namedWindow("test_eye");
        
        Mat im; resize(frame,im,Size(frame.cols/2,frame.rows/2));
        flip(im, im,1);
        Mat gray; cvtColor(im,gray,CV_RGB2GRAY);
        
        if(!init){
            vector<Point2f> eyes = fdet.findEyes(gray);
            if(eyes.size() != 2)continue;
            vector<Point2f> p(2);
            
            {
                cv::Mat t = gray.clone();
                cv::circle(t, eyes[0], 2, cv::Scalar(255), -1);
                cv::circle(t, eyes[1], 2, cv::Scalar(255), -1);
                //cv::imshow("test_eye", t);
                cv::waitKey(2);
            }
            
            
            p[0].x = (mean.db(37  )+mean.db(38  )+mean.db(40  )+mean.db(41  ))/4;
            p[0].y = (mean.db(37+n)+mean.db(38+n)+mean.db(40+n)+mean.db(41+n))/4;
            p[1].x = (mean.db(43  )+mean.db(44  )+mean.db(46  )+mean.db(47  ))/4;
            p[1].y = (mean.db(43+n)+mean.db(44+n)+mean.db(46+n)+mean.db(47+n))/4;
            double scale = fabs(eyes[1].x-eyes[0].x)/fabs(p[1].x-p[0].x);
            Mat plocal = Mat::zeros(pdm.nModes(),1,CV_64F);
            Mat pglobl = (Mat_<double>(6,1) <<
                          scale,0,0,0,eyes[0].x-scale*p[0].x,eyes[0].y-scale*p[0].y);
            pdm.CalcShape2D(s,plocal,pglobl);
            r = stacked2inter(s);
            r = R.predict(gray,r,R._niter);
            init = true;
        }else{
            int samples = 5;
            vector<Mat_<float> > rr(samples); Mat_<float> dr(2*n,1),r0 = r.clone();
            gettimeofday(&t0, NULL);
            for(int i = 0; i < samples; i++){
                rn.fill(dr,RNG::UNIFORM,-3,3); r = r0 + dr;
                s = inter2stacked(r);
                pdm.CalcParams(s,plocal,pglobl);
                pdm.Clamp(plocal,2.0);
                pdm.CalcShape2D(s,plocal,pglobl);
                r = stacked2inter(s);
                rr[i] = R.predict(gray,r);
            }
            r = rr[0].clone();
            for(int i = 1; i < samples; i++)r += rr[i];
            r /= samples; dr = 0.0;
            for(int i = 0; i < samples; i++)dr += (rr[i]-r).mul(rr[i]-r);
            vector<int> size(n);
            for(int i = 0; i < n; i++){
                float vx = dr(2*i),vy = dr(2*i+1);
                size[i] = max(1.0f,3*sqrt((vx+vy)/(2*samples+1)));
            }
            
            gettimeofday(&t1,NULL);
            
            //for(int i = 0; i < samples; i++)
            //drawPts(im,rr[i],CV_RGB(255,0,0));
            drawShape(im,r,con,CV_RGB(255,0,0),CV_RGB(0,255,0));//,size);
            
            std::cout << "FPS: "<< (1e3*(t1.tv_sec-t0.tv_sec) + 1e-3*(t1.tv_usec - t0.tv_usec)) << std::endl;
        }
        /*
         s = inter2stacked(r);
         pdm.CalcParams(s,plocal,pglobl);
         pdm.Clamp(plocal,4.0);
         pdm.CalcShape2D(s,plocal,pglobl);
         r = stacked2inter(s);
         drawShape(im,r,con,CV_RGB(255,0,0),CV_RGB(255,0,0));
         */
        imshow("tracking",im);
        int c = waitKey(10); if(c == 27)break; else if(c == 'd')init = false;
    }return;
}
//=========================================================================
int main()
{
    train();
    //test();
    //test_track();
    return 0;
}
//=========================================================================
