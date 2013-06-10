/**
 shapeRegressor: Random-ferns shape regressor
 Jason Saragih (2012)
 */
#include "shapeRegressor.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>
//=========================================================================
void
fern::
write(FILE* f)
{
    fwrite(&_npts,sizeof(int),1,f);
    fwrite(&_nfeats,sizeof(int),1,f);
    fwrite(_xloc.ptr<float>(0),sizeof(float),2*_nfeats,f);
    fwrite(_yloc.ptr<float>(0),sizeof(float),2*_nfeats,f);
    fwrite(_thresh.ptr<float>(0),sizeof(float),_nfeats,f);
    fwrite(_index.ptr<int>(0),sizeof(int),2*_nfeats,f);
    int nbins = (int)pow(2.0,_nfeats);
    for(int i = 0; i < nbins; i++)
        fwrite(_dshape[i].ptr<float>(0),sizeof(float),2*_npts,f);
    return;
}
//=========================================================================
void
fern::
read(FILE* f)
{
    fread(&_npts,sizeof(int),1,f);
    fread(&_nfeats,sizeof(int),1,f);
    _xloc.create(_nfeats,2); _yloc.create(_nfeats,2);
    _thresh.create(_nfeats,1); _index.create(_nfeats,2);
    fread(_xloc.ptr<float>(0),sizeof(float),2*_nfeats,f);
    fread(_yloc.ptr<float>(0),sizeof(float),2*_nfeats,f);
    fread(_thresh.ptr<float>(0),sizeof(float),_nfeats,f);
    fread(_index.ptr<int>(0),sizeof(int),2*_nfeats,f);
    int nbins = (int)pow(2.0,_nfeats); _dshape.resize(nbins);
    for(int i = 0; i < nbins; i++){
        _dshape[i].create(2*_npts,1);
        fread(_dshape[i].ptr<float>(0),sizeof(float),2*_npts,f);
    }this->memAlloc(); return;
}
//=========================================================================
vector<Mat_<float> >
fern::
train(const int nfeats, const float beta,
      const vector<Mat_<float> > &dshapes,
      const Mat_<float> &xyloc,
      const Mat_<float> &values,
      const Mat_<int> &index,
      const Mat_<float> &cov)
{
    cout << "fern train function is called" << endl;
    int npts = dshapes[0].rows/2,nsamples = dshapes.size(),ncand = index.rows;
    vector<int> idx1(nfeats),idx2(nfeats); RNG rn(getTickCount());
    _npts = npts; _nfeats = nfeats; this->memAlloc();
    _xloc.create(_nfeats,2); _yloc.create(_nfeats,2);
    _thresh.create(_nfeats,1); _index.create(_nfeats,2);
    
    for(int f = 0; f < nfeats; f++){
        Mat_<float> d(2*npts,1),y(nsamples,1);
        rn.fill(d,RNG::UNIFORM,-1,1); normalize(d,d);
        for(int i = 0; i < nsamples; i++)y(i) = d.dot(dshapes[i]);
        int i1,i2;
        if(!cov.empty()){
            Mat_<float> vy = values*y; int i1_best=-1,i2_best=-1; float v_best=-1;
            for(int c1 = 0; c1 < ncand; c1++){
                for(int c2 = c1+1; c2 < ncand; c2++){
                    float v = fabs((vy(c1)-vy(c2))/(cov(c1,c1)-2*cov(c1,c2)+cov(c2,c2)));
                    if(cvIsNaN(v) || cvIsInf(v))v = 0;
                    if((i1_best < 0) || (v > v_best)){
                        i1_best = c1; i2_best = c2; v_best = v;
                    }
                }
            }
            i1 = i1_best; i2 = i2_best;
        }else{ i1 = rn.uniform(0,ncand);
            while(1){i2 = rn.uniform(0,ncand); if(i2 != i1)break;}
        }
        Mat_<float> dv = values.row(i1) - values.row(i2);
        float best_thresh = -1,best_var = -1;
        for(int i = 0; i < 100; i++){
            float thresh = dv(rn.uniform(0,nsamples)); int n1 = 0,n2 = 0;
            float m1 = 0,m2 = 0;
            float *dp = dv.ptr<float>(0),*yp = y.ptr<float>(0);
            for(int j = 0; j < nsamples; j++,dp++,yp++){
                if(*dp >= thresh){m1 += *yp; n1++;}
                else{m2 += *yp; n2++;}
            }
            m1 /= n1; m2 /= n2;
            float v1 = 0,v2 = 0; dp = dv.ptr<float>(0); yp = y.ptr<float>(0);
            for(int j = 0; j < nsamples; j++,dp++,yp++){
                if(*dp >= thresh)v1 += (*yp-m1)*(*yp-m1);
                else v2 += (*yp-m2)*(*yp-m2);
            }
            float v = n1*log(v1/n1+1e-6) + n2*log(v2/n2+1e-6);
            if((best_var < 0) || (best_var > v)){
                best_var = v; best_thresh = thresh;
            }
        }
        _xloc(f,0) = xyloc(i1,0); _xloc(f,1) = xyloc(i2,0);
        _yloc(f,0) = xyloc(i1,1); _yloc(f,1) = xyloc(i2,1);
        _index(f,0) = index(i1); _index(f,1) = index(i2);
        _thresh(f) = best_thresh; idx1[f] = i1; idx2[f] = i2;
    }
    /*
     for(int f = 0; f < nfeats; f++){
     Mat_<float> d(2*npts,1),y(nsamples,1);
     rn.fill(d,RNG::UNIFORM,-1,1); normalize(d,d);
     for(int i = 0; i < nsamples; i++)y(i) = d.dot(dshapes[i]);
     int i1_best,i2_best; float best_thresh,best_var = -1;
     for(int k = 0; k < 100; k++){
     int i1 = rn.uniform(0,ncand),i2;
     while(1){i2 = rn.uniform(0,ncand); if(i2 != i1)break;}
     Mat_<float> dv = values.row(i1) - values.row(i2);
     for(int i = 0; i < 100; i++){
     float thresh = dv(rn.uniform(0,nsamples)); int n1 = 0,n2 = 0;
     float m1 = 0,m2 = 0; float *dp = dv.ptr<float>(0),*yp = y.ptr<float>(0);
     for(int j = 0; j < nsamples; j++,dp++,yp++){
     if(*dp >= thresh){m1 += *yp; n1++;} else{m2 += *yp; n2++;}
     }
     m1 /= n1; m2 /= n2;
     float v1 = 0,v2 = 0; dp = dv.ptr<float>(0); yp = y.ptr<float>(0);
     for(int j = 0; j < nsamples; j++,dp++,yp++){
     if(*dp >= thresh)v1 += (*yp-m1)*(*yp-m1); else v2 += (*yp-m2)*(*yp-m2);
     }
     float v = n1*log(v1/n1+1e-6) + n2*log(v2/n2+1e-6);
     if((best_var < 0) || (best_var > v)){
     best_var = v; best_thresh = thresh; i1_best = i1; i2_best = i2;
     }
     }
     }
     _xloc(f,0) = xyloc(i1_best,0); _xloc(f,1) = xyloc(i2_best,0);
     _yloc(f,0) = xyloc(i1_best,1); _yloc(f,1) = xyloc(i2_best,1);
     _index(f,0) = index(i1_best); _index(f,1) = index(i2_best);
     _thresh(f) = best_thresh; idx1[f] = i1_best; idx2[f] = i2_best;
     }
     */
    int nbins = (int)pow(2.0,_nfeats); vector<int> bidx(nsamples);
    vector<int> dsum(nbins); _dshape.resize(nbins);
    for(int i = 0; i < nbins; i++){
        dsum[i] = 0; _dshape[i] = Mat_<float>::zeros(2*npts,1);
    }
    for(int i = 0; i < nsamples; i++){
        int idx = 0,v = 1;
        for(int j = 0; j < _nfeats; j++,v *= 2){
            int i1 = idx1[j],i2 = idx2[j];
            idx += v*int((float)values(i1,i) - (float)values(i2,i) >= _thresh(j));
        }
        _dshape[idx] += dshapes[i]; dsum[idx] += 1; bidx[i] = idx;
    }
    for(int i = 0; i < nbins; i++){
        if(dsum[i] == 0)_dshape[i] = 0.0f;
        else _dshape[i] /= (1.0f + beta/(dsum[i]+1e-8))*dsum[i];
    }
    vector<Mat_<float> > residuals(nsamples);
    for(int i = 0; i < nsamples; i++)residuals[i] = dshapes[i]-_dshape[bidx[i]];
    return residuals;
    
    
    // int npts = dshapes[0].rows/2,nsamples = dshapes.size(),ncand = index.rows;
    // vector<int> idx1(nfeats),idx2(nfeats); RNG rn(getTickCount());
    // for(int f = 0; f < nfeats; f++){
    //   /*
    //   Mat_<float> d(2*npts,1),y(nsamples,1);
    //   rn.fill(d,RNG::UNIFORM,-1,1); normalize(d,d);
    //   for(int i = 0; i < nsamples; i++)y(i) = d.dot(dshapes[i]);
    //   Mat_<float> vy = values*y; int i1_best=-1,i2_best=-1; float v_best=-1;
    //   for(int i1 = 0; i1 < ncand; i1++){
    //     for(int i2 = i1+1; i2 < ncand; i2++){
    //   float v = fabs((vy(i1)-vy(i2))/(cov(i1,i1)-2*cov(i1,i2)+cov(i2,i2)));
    // 	if(cvIsNaN(v) || cvIsInf(v))v = 0;
    // 	if((i1_best < 0) || (v > v_best)){
    // 	  i1_best = i1; i2_best = i2; v_best = v;
    // 	}
    //     }
    //   }
    //   idx1[f] = i1_best; idx2[f] = i2_best;
    //   */
    //   idx1[f] = rn.uniform(0,ncand);
    //   idx2[f] = rn.uniform(0,ncand);
    // }
    // _npts = npts; _nfeats = nfeats; this->memAlloc();
    // _xloc.create(_nfeats,2); _yloc.create(_nfeats,2);
    // _thresh.create(_nfeats,1); _index.create(_nfeats,2);
    // for(int i = 0; i < nfeats; i++){
    //   int i1 = idx1[i],i2 = idx2[i];
    //   _xloc(i,0) = xyloc(i1,0); _xloc(i,1) = xyloc(i2,0);
    //   _yloc(i,0) = xyloc(i1,1); _yloc(i,1) = xyloc(i2,1);
    //   _index(i,0) = index(i1); _index(i,1) = index(i2);
    //   _thresh(i) =
    //     values(i1,rn.uniform(0,nsamples)) -
    //     values(i2,rn.uniform(0,nsamples));
    // }
    // int nbins = (int)pow(2.0,_nfeats); vector<int> bidx(nsamples);
    // vector<int> dsum(nbins); _dshape.resize(nbins);
    // for(int i = 0; i < nbins; i++){
    //   dsum[i] = 0; _dshape[i] = Mat_<float>::zeros(2*npts,1);
    // }
    // for(int i = 0; i < nsamples; i++){ int idx = 0,v = 1;
    //   for(int j = 0; j < _nfeats; j++,v *= 2){ int i1 = idx1[j],i2 = idx2[j];
    //     idx += v*int((float)values(i1,i) - (float)values(i2,i) >= _thresh(j));
    //   }
    //   _dshape[idx] += dshapes[i]; dsum[idx] += 1; bidx[i] = idx;
    // }
    // for(int i = 0; i < nbins; i++){
    //   if(dsum[i] == 0)_dshape[i] = 0.0f;
    //   else _dshape[i] /= (1.0f + beta/(dsum[i]+1e-8))*dsum[i];
    // }
    // vector<Mat_<float> > residuals(nsamples);
    // for(int i = 0; i < nsamples; i++)residuals[i] = dshapes[i]-_dshape[bidx[i]];
    // return residuals;
}
//=========================================================================
Mat_<float>
fern::
predict(const Mat_<uchar> &im,const Mat_<float> &pt)
{
    assert(pt.rows == 2*_npts);
    for(int i = 0; i < _nfeats; i++){
        int j1 = _index(i,0),j2 = _index(i,1);
        xmap_(i,0) = _xloc(i,0) + pt(2*j1  );
        xmap_(i,1) = _xloc(i,1) + pt(2*j2  );
        ymap_(i,0) = _yloc(i,0) + pt(2*j1+1);
        ymap_(i,1) = _yloc(i,1) + pt(2*j2+1);
    }
    remap(im,feats_,xmap_,ymap_,CV_INTER_LINEAR); int idx = 0;
    for(int i = 0, v=1; i < _nfeats; i++,v *= 2)
        idx += v*int((float)feats_(i,0) - (float)feats_(i,1) >= _thresh(i));
    return _dshape[idx].clone();
}
//=========================================================================
void
fern::
memAlloc()
{
    xmap_.create(_nfeats,2);
    ymap_.create(_nfeats,2);
    feats_.create(_nfeats,2);
}
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
void
fernCascade::
write(FILE* f)
{
    fwrite(&_nlevels,sizeof(int),1,f);
    for(int i = 0; i < _nlevels; i++)_F[i].write(f);
    return;
}
//=========================================================================
void
fernCascade::
read(FILE* f)
{
    fread(&_nlevels,sizeof(int),1,f); _F.resize(_nlevels);
    for(int i = 0; i < _nlevels; i++)_F[i].read(f);
    return;
}
//=========================================================================
void
fernCascade::
train(const int nlevels,const int nfeats, const float beta, const int c_iter,
      const vector<Mat_<float> > &shapes,
      const vector<Mat_<float> > &target,
      const vector<Mat_<uchar> > &images,
      const Mat_<float> &xyloc,
      const Mat_<int> &index)
{
    cout << "fernCascade train function is called " << endl;
    int nsamples = shapes.size(),ncand = index.rows;
    Mat_<float> xmap(ncand,1),ymap(ncand,1); Mat_<uchar> crop(ncand,1);
    Mat_<float> values(ncand,nsamples),val(ncand,1);
    for(int i = 0; i < nsamples; i++){
        for(int j = 0; j < ncand; j++){
            int k = index(j);
            xmap(j) = xyloc(j,0) + shapes[i](2*k  );
            ymap(j) = xyloc(j,1) + shapes[i](2*k+1);
        }
        remap(images[i],crop,xmap,ymap,CV_INTER_LINEAR);
        crop.convertTo(val,CV_32F); Mat vi = values.col(i); val.copyTo(vi);
    }
    Mat_<float> cov = values*values.t();
    vector<Mat_<float> > dshapes(nsamples);
    for(int i = 0; i < nsamples; i++)dshapes[i] = target[i] - shapes[i];
    _nlevels = nlevels; _F.resize(nlevels);
    
    // open txt file to save the error
    char file_name [100];
    sprintf(file_name, "/Users/Jun/Documents/Train_error%d.txt", c_iter);
    ofstream poi1;
    poi1.open(file_name,ios::out|ios::app);
    poi1.setf(ios::fixed, ios::floatfield);
    poi1.setf(ios::showpoint);
    // end open file
    
    for(int i = 0; i < nlevels; i++){
        //*********
        double e = 0.0;
        for(int j = 0; j < nsamples; j++)
            e += norm(dshapes[j]);
        cout << i << " " << e << endl;
        
        poi1<< e <<  endl;
        //*********
        dshapes = _F[i].train(nfeats,beta,dshapes,xyloc,values,index,cov);
    }return;
    // close the file
    poi1.close();
}
//=========================================================================
Mat_<float>
fernCascade::
predict(const Mat_<uchar> &im,const Mat_<float> &pt)
{
    Mat_<float> s = pt.clone();
    for(int i = 0; i < _nlevels; i++)s += _F[i].predict(im,pt);
    return s-pt;
}
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
//=========================================================================
int
shapeRegressor::
save(const char* fname)
{
    FILE *f = fopen(fname,"wb"); if(f == NULL)return -1;
    this->write(f); return 0;
}
//=========================================================================
int
shapeRegressor::
load(const char* fname)
{
    FILE *f = fopen(fname,"rb"); if(f == NULL)return -1;
    this->read(f); return 0;
}
//=========================================================================
void
shapeRegressor::
write(FILE* f)
{
    int n = _ref.rows;
    fwrite(&_niter,sizeof(int),1,f);
    fwrite(&_size.width,sizeof(int),1,f);
    fwrite(&_size.height,sizeof(int),1,f);
    fwrite(&n,sizeof(int),1,f);
    fwrite(_ref.ptr<float>(0),sizeof(float),n,f);
    for(int i = 0; i < _niter; i++)_C[i].write(f);
    return;
}
//=========================================================================
void
shapeRegressor::
read(FILE* f)
{
    fread(&_niter,sizeof(int),1,f);
    fread(&_size.width,sizeof(int),1,f);
    fread(&_size.height,sizeof(int),1,f);
    int n; fread(&n,sizeof(int),1,f); _ref.create(n,1);
    fread(_ref.ptr<float>(0),sizeof(float),n,f); _C.resize(_niter);
    for(int i = 0; i < _niter; i++)_C[i].read(f);
    return;
}
//=========================================================================
void
shapeRegressor::
train(const vector<string> &imlist,
      const vector<Mat_<float> > &landmarks,
      const Mat_<float> &ref,
      const Size &size,
      const int niter,
      const int nlevels,
      const int nfeats,
      const int ncand,
      const int nsamples,
      const float beta,
      const float pert)
{
    cout << "shapeRegressor train function is called " << endl;
    _niter = niter; _size = size; _ref = ref.clone(); _C.resize(niter);
    RNG rn(getTickCount()); int n = landmarks[0].rows/2;
    for(int iter = 0; iter < niter; iter++){
        vector<Mat_<float> > shapes(nsamples);
        vector<Mat_<float> > target(nsamples);
        vector<Mat_<uchar> > images(nsamples);
        for(int i = 0; i < nsamples; i++){
            
            int idx = rn.uniform(0,imlist.size());
            //********************************************
            //Mat_<uchar> im = this->norm_illum(imread(imlist[idx],0));
            //if(this->hasNans(im)){cout << "NaN!" << endl; exit(0);}//////////////////
            Mat_<uchar> im = imread(imlist[idx],0);
            //********************************************
            Mat_<float> pt = landmarks[idx].clone(),dp(2*n,1);
            Mat_<float> shape = _ref.clone();
            {
                float mx = 0,my = 0;
                for(int j = 0; j < n; j++){mx += shape(2*j); my += shape(2*j+1);}
                mx /= n; my /= n;
                for(int j = 0; j < n; j++){shape(2*j) -= mx; shape(2*j+1) -= my;}
                Rect bb = this->calcBoundingBox(_ref);
                float dx = rn.uniform(-bb.width *pert,bb.width *pert) + mx;
                float dy = rn.uniform(-bb.height*pert,bb.height*pert) + my;
                float theta = rn.uniform(-CV_PI/20,CV_PI/20);
                float scale = rn.uniform(0.9,1.1);
                float a = scale*cos(theta),b = scale*sin(theta);
                for(int j = 0; j < n; j++){
                    float x = shape(2*j),y = shape(2*j+1);
                    shape(2*j  ) = a*x - b*y + dx;
                    shape(2*j+1) = b*x + a*y + dy;
                }
            }

            
            float a1,b1,x1,y1; this->calcSimil(shape,pt,a1,b1,x1,y1);
            {
                Mat_<float> p = landmarks[rn.uniform(0,imlist.size())].clone();
                float a,b,x,y; this->calcSimil(_ref,p,a,b,x,y);
                float ai,bi,xi,yi; this->invSimil(a,b,x,y,ai,bi,xi,yi);
                for(int j = 0; j < n; j++){
                    float vx = ai*p(2*j) - bi*p(2*j+1) + xi;
                    float vy = bi*p(2*j) + ai*p(2*j+1) + yi;
                    shape(2*j  ) = a1*vx - b1*vy + x1;
                    shape(2*j+1) = b1*vx + a1*vy + y1;
                }
                shape = this->predict(im,shape,iter);
            }
            this->calcSimil(_ref,shape,a1,b1,x1,y1);
            float a2,b2,x2,y2; this->invSimil(a1,b1,x1,y1,a2,b2,x2,y2);
            int ksize = int(round(a1/cos(atan2(b1,a1)))); if(ksize%2 == 0)ksize+=1;
            Mat M = (Mat_<float>(2,3) << a1,-b1,x1,b1,a1,y1);
            if(ksize > 1){
                Mat_<uchar> smooth; GaussianBlur(im,smooth,Size(ksize,ksize),0,0);
                warpAffine(smooth,images[i],M,_size,WARP_INVERSE_MAP+INTER_LINEAR);
            }else{
                warpAffine(im,images[i],M,_size,WARP_INVERSE_MAP+INTER_LINEAR);
            }
            target[i].create(2*n,1); shapes[i].create(2*n,1);
            for(int j = 0; j < n; j++){
                target[i](2*j  ) = a2*pt(2*j) - b2*pt(2*j+1) + x2;
                target[i](2*j+1) = b2*pt(2*j) + a2*pt(2*j+1) + y2;
                shapes[i](2*j  ) = a2*shape(2*j) - b2*shape(2*j+1) + x2;
                shapes[i](2*j+1) = b2*shape(2*j) + a2*shape(2*j+1) + y2;
            }
            //***************
            {
                Mat img; cvtColor(images[i],img,CV_GRAY2RGB);
                for(int l = 0; l < 66; l++){
                    circle(img,Point2f(target[i](2*l),target[i](2*l+1)),1,
                           CV_RGB(0,255,0),1,CV_AA);
                    circle(img,Point2f(shapes[i](2*l),shapes[i](2*l+1)),1,
                           CV_RGB(255,0,0),1,CV_AA);
                }
                imshow("test",img); if(waitKey(10) == 27) break;
            }
            //***************
        }
        
        Rect bb = this->calcBoundingBox(_ref);
        int dx = (_size.width-bb.width)/4,dy = (_size.height-bb.height)/4;
        Mat_<float> xyloc(ncand,2); Mat_<int> index(ncand,1);
        for(int i = 0; i < ncand; i++){
            Point2f p(rn.uniform(dx,_size.width- dx),
                      rn.uniform(dy,_size.height-dy));
            index(i) = this->findClosestPoint(p,_ref);
            xyloc(i,0) = p.x - _ref(2*index(i)  );
            xyloc(i,1) = p.y - _ref(2*index(i)+1);
        }
        _C[iter].train(nlevels,nfeats,beta,iter,shapes,target,images,xyloc,index); //comment for the fernCascade train
    }return;
}
//=========================================================================
Mat_<float>
shapeRegressor::
predict(const Mat_<uchar> &im,const Mat_<float> &pt,
        const int niter)
{
    
    //*********
    //Mat img = this->norm_illum(im);
    //*********
    
    
    Mat_<float> s = pt.clone(); int n = s.rows/2;
    int nIter = niter; if((nIter < 0) || (nIter >= _niter))nIter = _niter;
    for(int iter = 0; iter < nIter; iter++){
        Mat_<float> M = this->cropImagePts(im,s,crop_,s); ////////////////////
        //Mat_<float> M = this->cropImagePts(img,s,crop_,s);
        s += _C[iter].predict(crop_,s);
        for(int j = 0; j < n; j++){
            float x = s(2*j),y = s(2*j+1);
            s(2*j  ) = M(0,0)*x + M(0,1)*y + M(0,2);
            s(2*j+1) = M(1,0)*x + M(1,1)*y + M(1,2);
        }
    }return s;
}
//=========================================================================
int
shapeRegressor::
findClosestPoint(const Point2f &p,const Mat_<float> &pt)
{
    float vmin = -1; int imin = -1,n = pt.rows/2;
    for(int i = 0; i < n; i++){
        float vx = p.x - pt(2*i  );
        float vy = p.y - pt(2*i+1);
        float v = vx*vx + vy*vy;
        if((imin < 0) || (vmin > v)){vmin = v; imin = i;}
    }return imin;
}
//=========================================================================
Rect
shapeRegressor::
calcBoundingBox(const Mat_<float> &pt)
{
    int n = pt.rows/2;
    float xmin = pt(0),xmax = pt(0),ymin = pt(1),ymax = pt(1);
    for(int i = 1; i < n; i++){
        xmin = min(xmin,pt(2*i  ));
        xmax = max(xmax,pt(2*i  ));
        ymin = min(ymin,pt(2*i+1));
        ymax = max(ymax,pt(2*i+1));
    }return Rect(xmin,ymin,xmax-xmin+1,ymax-ymin+1);
}
//=========================================================================
Mat_<float>
shapeRegressor::
cropImagePts(const Mat_<uchar> &im,const Mat_<float> &pt,
             Mat_<uchar> &crop,Mat_<float> &s)
{
    float a1,b1,x1,y1; this->calcSimil(_ref,pt,a1,b1,x1,y1);
    float a2,b2,x2,y2; this->invSimil(a1,b1,x1,y1,a2,b2,x2,y2);
    int ksize = int(round(a1/cos(atan2(b1,a1)))); if(ksize % 2 == 0)ksize+= 1;
    Mat M = (Mat_<float>(2,3) << a1,-b1,x1,b1,a1,y1);
    if(ksize > 1){
        Mat_<uchar> smooth; GaussianBlur(im,smooth,Size(ksize,ksize),0,0);
        warpAffine(smooth,crop,M,_size,WARP_INVERSE_MAP+INTER_LINEAR);
    }else{
        warpAffine(im,crop,M,_size,WARP_INVERSE_MAP+INTER_LINEAR);
    }
    int n = pt.rows/2; if(s.rows != 2*n)s.create(2*n,1);
    for(int j = 0; j < n; j++){
        float x = pt(2*j),y = pt(2*j+1);
        s(2*j  ) = a2*x - b2*y + x2;
        s(2*j+1) = b2*x + a2*y + y2;
    }return M;
}
//=========================================================================
void
shapeRegressor::
calcSimil(const Mat_<float> &src,const Mat_<float> &dst,
          float &a,float &b,float &tx,float &ty)
{
    Mat_<float> H = Mat_<float>::zeros(4,4),g = Mat_<float>::zeros(4,1),p(4,1);
    for(int i = 0; i < src.rows/2; i++){
        float x1 = src(2*i),y1 = src(2*i+1);
        float x2 = dst(2*i),y2 = dst(2*i+1);
        H(0,0) += x1*x1 + y1*y1; H(0,2) += x1; H(0,3) += y1;
        g(0,0) += x1*x2 + y1*y2; g(1,0) += x1*y2 - y1*x2;
        g(2,0) += x2; g(3,0) += y2;
    }
    H(1,1) = H(0,0); H(1,2) = H(2,1) = -1.0*(H(3,0) = H(0,3));
    H(1,3) = H(3,1) = H(2,0) = H(0,2); H(2,2) = H(3,3) = src.rows/2;
    solve(H,g,p,DECOMP_CHOLESKY);
    a = p(0,0); b = p(1,0); tx = p(2,0); ty = p(3,0); return;
}
//=========================================================================
void
shapeRegressor::
invSimil(float a1,float b1,float tx1,float ty1,
         float& a2,float& b2,float& tx2,float& ty2)
{
    Mat_<float> M = (cv::Mat_<float>(2,2) << a1, -b1, b1, a1);
    Mat_<float> N = M.inv(cv::DECOMP_SVD); a2 = N(0,0); b2 = N(1,0);
    tx2 = -1.0*(N(0,0)*tx1 + N(0,1)*ty1);
    ty2 = -1.0*(N(1,0)*tx1 + N(1,1)*ty1); return;
}
//=========================================================================
Mat_<uchar>
shapeRegressor::
norm_illum(const Mat_<uchar> &img,
           const Size size,
           const float thresh)
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
bool
shapeRegressor::
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
    }else if(X.type() == CV_8U){
        const uchar* xp = X.ptr<uchar>(0); int n = X.rows*X.cols;
        for(int i = 0; i < n; i++,++xp){
            if(cvIsNaN(*xp))return true;
            if(cvIsInf(*xp))return true;
        }return false;
    }else return true;
}
//=========================================================================  
