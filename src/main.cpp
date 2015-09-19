#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "daisy/daisy.h"
#include "math.h"
#define __STDC_FORMAT_MACROS  //define this to make intypes.h works
#include "stdint.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"
using namespace tbb;

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <windows.h>
//Definition de constante de paramétrisation de la caméra
#define FRAME_WIDTH 320
#define FRAME_HEIGHT 240

CvCapture *cameraLeft, *cameraRight;

struct distortionMapsStruct {
	CvMat* mx1;
    CvMat* my1;
    CvMat* mx2; 
    CvMat* my2;
};

struct configStruct {
	bool parseOK;
	int board_w;
    int board_h;
    int n_boards;
	bool calib;
};

distortionMapsStruct computeDistortionMapping(CvSize frameLeftSize, CvSize frameRightSize);
void computeSingleCalibration(int nbrImage,int board_n,CvMat* image_points,CvMat* object_points,CvMat* point_counts, CvSize imgSize,char *filePrefix);
void computeStereoCalibration(CvMat* image_points_left,CvMat* object_points_left,CvMat* point_counts_left, CvSize frameLeftSize,
	CvMat* image_points_right,CvMat* object_points_right,CvMat* point_counts_right,CvSize frameRightSize);
void startStereoMatching(CvCapture* cameraLeft,CvCapture* cameraRight);
void startStereoCalibration(configStruct config, CvCapture* cameraLeft,CvCapture* cameraRight);
	 

//NumberOfBinInHistogram=histq
//NumberOfHistogramPerLayer = thq
//NumberOfConvolLayer=radq
//NumberOfHistogram=(*NumberOfHistogramPerLayer)*(*NumberOfConvolLayer)+1;
class ComputeLeftDisparity{
    float *dzyL, *dzyR;
    int16_t *disparityL,width, height,dimDzy,dispMin,dispMax;
    int16_t rad, radq, thq, histq;
public :
    void operator( ) (const blocked_range<size_t> &r)const
    {
        int64_t offset,offsetR;
        int32_t XL,YL,XR,k,XRMin,xstart,xstop;
        int32_t h;
        uint32_t offsethMassL,offsethMassR;
        float hMass;
        float DzyMinMass,DzyMass;
        for(YL=r.begin();YL<r.end();YL++)
        {
            for(XL=dispMin;XL<width-dispMin;XL++)
            {
                offset=width*YL+XL;

                    if(XL-dispMax<0)
                    {
                        xstart=0;
                    }else{
                        xstart=XL-dispMax;
                    }

                    xstop=XL-dispMin;

                    XRMin=0;

                    for(XR=xstart;XR<=xstop;XR++)
                    {
                        offsetR=width*YL+XR;
                        //for each histogram compute euclidian distance
                        DzyMass=(float) 0;
                        for(k=0;k<thq*radq+1;k++)
                        {
                            hMass=0;
                            offsethMassL=offset*dimDzy+k*histq;
                            offsethMassR=offsetR*dimDzy+k*histq;
                            for(h=0;h<histq;h++)
                            {
                                hMass=hMass+pow((float)(dzyL[offsethMassL+h]-dzyR[offsethMassR+h]),(int)2);
                            }
                            DzyMass=DzyMass+hMass;
                        }


                        if(XR==xstart)
                        {
                            DzyMinMass=DzyMass;
                        }

                        if((float)DzyMass<(float)DzyMinMass)
                        {
                            DzyMinMass=DzyMass;
                            XRMin=XR;
                        }

                    }
                    if(  ((XL-XRMin) < dispMax ) & ((XL-XRMin) > dispMin ) )
                    {
                        disparityL[offset]=(int16_t) (XL-XRMin);
                    }else{
                        disparityL[offset]=(int16_t) -1;
                    }

            }
        }

    }
    ComputeLeftDisparity(float *_dzyL,float *_dzyR,int16_t *_disparityL,int16_t _width, int16_t _height,int16_t _dispMin,int16_t _dispMax,int16_t _rad, int16_t _radq,int16_t _thq,int16_t _histq)
       {
        dzyL=_dzyL;
        dzyR=_dzyR;
        dimDzy= (thq*radq+1)*histq;
        disparityL=_disparityL;
        width=_width;
        height=_height;
        dispMin=_dispMin;
        dispMax=_dispMax;
        rad=_rad;
        radq=_radq;
        thq=_thq;
        histq=_histq;
    }
};



void fnExit(void)
{
  	cvDestroyAllWindows();
	if(cameraLeft != NULL){
		cvReleaseCapture(&cameraLeft);
	}
	if(cameraRight != NULL){
		cvReleaseCapture(&cameraRight);
	}
}

void printArgsUsage(char* argv[]){
	fprintf(stderr,"USAGE: %s (board_w board_h n_boards)\n",argv[0]);
	fprintf(stderr,"\t board_w (only for calibration) : chessboard width. Example : 9\n");
    fprintf(stderr,"\t board_h (only for calibration) : chessboard height. Example : 6\n");
    fprintf(stderr,"\t n_boards (only for calibration) : Number of chessboard required for calibration. Example : 10\n");
}

configStruct parseArgs(int argc, char* argv[]){
	
	configStruct config;
    config.parseOK=false;

	if(argc ==1 ){
		config.calib=false;
		config.parseOK=true;
	}else if(argc ==4 ){
		config.board_w = atoi(argv[1]);
		config.board_h = atoi(argv[2]);
		config.n_boards = atoi(argv[3]);
		config.calib=true;
		config.parseOK=true;
	}else{
		printArgsUsage(argv);
	}
	
	return config;
}



int main(int argc, char* argv[]) {
	/*try{
		std::cout<<cv::gpu::getDevice();
	}catch(const cv::Exception& ex){
		std::cout << "Error: " << ex.what() <<std::endl;
	}
	return 0;
*/
	configStruct config = parseArgs(argc, argv);
	if(config.parseOK == false){
		return EXIT_FAILURE;
	}
    
	atexit (fnExit);

    //On va récupérer la webcam grace à DirectShow (CV_CAP_DSHOW)
    cameraLeft = cvCreateCameraCapture( 0 );
	
    if( !cameraLeft)
    {
        printf("Erreur : stereo camera Left not found\n");
        return EXIT_FAILURE;
    }

	cameraRight = cvCreateCameraCapture(CV_CAP_DSHOW+ 1 );
	if( !cameraRight)
    {
		cvReleaseCapture(&cameraLeft);
        printf("Erreur : stereo camera Right not found\n");
        return EXIT_FAILURE;
    }

	Sleep(200);
	 //On paramètre la caméra selon les valeurs définis en entête FRAME_WIDTH et FRAME_HEIGHT pour la taille de l'image capturée

	cvSetCaptureProperty(cameraLeft, CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH  );
    cvSetCaptureProperty(cameraLeft, CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT  );
	cvSetCaptureProperty(cameraRight, CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH  );
    cvSetCaptureProperty(cameraRight, CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT  );


	printf("Parametrisation de la camera Left: \n\t width = %d\theight = %d\tfps = %f\n",(long)cvGetCaptureProperty(cameraLeft, CV_CAP_PROP_FRAME_WIDTH ) ,(long)cvGetCaptureProperty(cameraLeft, CV_CAP_PROP_FRAME_HEIGHT ),(float)cvGetCaptureProperty(cameraLeft, CV_CAP_PROP_FPS ));
	printf("Parametrisation de la camera Right: \n\t width = %d\theight = %d\tfps = %f\n",(long)cvGetCaptureProperty(cameraRight, CV_CAP_PROP_FRAME_WIDTH ) ,(long)cvGetCaptureProperty(cameraRight, CV_CAP_PROP_FRAME_HEIGHT ),(float)cvGetCaptureProperty(cameraRight, CV_CAP_PROP_FPS ));

	if(config.calib == true){
		startStereoCalibration(config,cameraLeft,cameraRight);
	}else{
		startStereoMatching(cameraLeft,cameraRight);
	}

	/*
    cvReleaseMat(&object_points_left);
    cvReleaseMat(&image_points_left);
    cvReleaseMat(&point_counts_left);

    // EXAMPLE OF LOADING THESE MATRICES BACK IN:
    CvMat *intrinsic = (CvMat*)cvLoad("Intrinsics_left.xml");
    CvMat *distortion = (CvMat*)cvLoad("Distortion_left.xml");
    // Build the undistort map that we will use for all
    // subsequent frames.
    //
    IplImage* mapx = cvCreateImage( frameLeftSize, IPL_DEPTH_32F, 1 );
    IplImage* mapy = cvCreateImage( frameLeftSize, IPL_DEPTH_32F, 1 );
    cvInitUndistortMap( intrinsic,distortion,mapx,mapy);
    // Just run the camera to the screen, now showing the raw and
    // the undistorted image.
    //
    cvNamedWindow( "Undistort" );
    while(frame_left) {
        IplImage *t = cvCloneImage(frame_left);
        //IplImage *tt = cvCloneImage(image);
        cvShowImage( "Calibration", frame_left ); // Show raw image

        cvRemap( t, frame_left, mapx, mapy ); // Undistort image
        cvShowImage("Undistort", frame_left); // Show corrected image
        //cvUndistort2(t,tt,intrinsic_matrix,distortion_coeffs);
        //cvShowImage("Undistort2", tt); // Show corrected image

        cvReleaseImage(&t);
        //cvReleaseImage(&tt);
        //Handle pause/unpause and ESC
        int c = cvWaitKey(15);
        if(c == 'p') {
            c = 0;
            while(c != 'p' && c != 27) {
                c = cvWaitKey(250);
            }
        }
        if(c == 27)
            break;
        frame_left = cvQueryFrame( cameraLeft );
    }
	*/


    return EXIT_SUCCESS;
	
}

void startStereoCalibration(configStruct config, CvCapture* cameraLeft,CvCapture* cameraRight){
	int successes = 0;
    int frame = 0;
	int corner_count_left,corner_count_right;
	int board_n = config.board_w * config.board_h;
    CvSize board_sz = cvSize( config.board_w, config.board_h );

	//ALLOCATE STORAGE
	CvMat* image_points_left = cvCreateMat(config.n_boards*board_n,2,CV_32FC1);
	CvMat* image_points_right = cvCreateMat(config.n_boards*board_n,2,CV_32FC1);
    CvMat* object_points_left = cvCreateMat(config.n_boards*board_n,3,CV_32FC1);
	CvMat* object_points_right = cvCreateMat(config.n_boards*board_n,3,CV_32FC1);
    CvMat* point_counts_left = cvCreateMat(config.n_boards,1,CV_32SC1);
	CvMat* point_counts_right = cvCreateMat(config.n_boards,1,CV_32SC1);
    CvMat* intrinsic_matrix_left = cvCreateMat(3,3,CV_32FC1);
	CvMat* intrinsic_matrix_right = cvCreateMat(3,3,CV_32FC1);
    CvMat* distortion_coeffs_left = cvCreateMat(5,1,CV_32FC1);
	CvMat* distortion_coeffs_right = cvCreateMat(5,1,CV_32FC1);
	CvPoint2D32f* corners_left = new CvPoint2D32f[ board_n ];
	CvPoint2D32f* corners_right = new CvPoint2D32f[ board_n ];

	IplImage *frame_left = cvQueryFrame( cameraLeft );
	IplImage *frame_right = cvQueryFrame( cameraRight );

	CvSize frameLeftSize = cvGetSize(frame_left);
	CvSize frameRightSize = cvGetSize(frame_right);

    IplImage *frame_left_gray = cvCreateImage(frameLeftSize,8,1);
	IplImage *frame_right_gray = cvCreateImage(frameRightSize,8,1);

    // CAPTURE CORNER VIEWS LOOP UNTIL WE’VE GOT n_boards
    // SUCCESSFUL CAPTURES (ALL CORNERS ON THE BOARD ARE FOUND)
    cvNamedWindow("Stitched Window", CV_WINDOW_AUTOSIZE); 
	IplImage *_stitchedImage; 

	printf("-------------------\n");
	printf("Calibration\n");
	printf("Press N to take a screenshot for calibration\n");

	bool takeScreenshot =false;

    while(successes < config.n_boards) {
		printf(".");
        if(takeScreenshot==true){
			//Find chessboard corners:
			int found_left = cvFindChessboardCorners( frame_left, board_sz, corners_left, &corner_count_left,
													CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS
													);
			int found_right = cvFindChessboardCorners( frame_right, board_sz, corners_right, &corner_count_right,
													CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS
													);
			if( corner_count_left == board_n && corner_count_right == board_n ) {
				printf("\nChessboard found\n");
				//Get Subpixel accuracy on those corners
				cvCvtColor(frame_left, frame_left_gray, CV_BGR2GRAY);
				cvCvtColor(frame_right, frame_right_gray, CV_BGR2GRAY);
				cvFindCornerSubPix(frame_left_gray, corners_left, corner_count_left,cvSize(11,11),cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
				cvFindCornerSubPix(frame_right_gray, corners_right, corner_count_right,cvSize(11,11),cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

				//Draw it
				cvDrawChessboardCorners(frame_left, board_sz, corners_left,corner_count_left, found_left);
				cvDrawChessboardCorners(frame_right, board_sz, corners_right,corner_count_right, found_right);

				char imageFileName[20]; 
				sprintf(imageFileName,"image_%d_left.jpg", successes); 
				cvSaveImage(imageFileName ,frame_left);
				sprintf(imageFileName,"image_%d_right.jpg", successes); 
				cvSaveImage(imageFileName ,frame_right);

				// If we got a good board, add it to our data
			
				int step = successes*board_n;
				for( int i=step, j=0; j<board_n; ++i,++j ) {
					CV_MAT_ELEM(*image_points_left, float,i,0) = corners_left[j].x;
					CV_MAT_ELEM(*image_points_left, float,i,1) = corners_left[j].y;
					CV_MAT_ELEM(*object_points_left,float,i,0) = j/config.board_w;
					CV_MAT_ELEM(*object_points_left,float,i,1) = j%config.board_w;
					CV_MAT_ELEM(*object_points_left,float,i,2) = 0.0f;
				}
				CV_MAT_ELEM(*point_counts_left, int,successes,0) = board_n;
			
				for( int i=step, j=0; j<board_n; ++i,++j ) {
					CV_MAT_ELEM(*image_points_right, float,i,0) = corners_right[j].x;
					CV_MAT_ELEM(*image_points_right, float,i,1) = corners_right[j].y;
					CV_MAT_ELEM(*object_points_right,float,i,0) = j/config.board_w;
					CV_MAT_ELEM(*object_points_right,float,i,1) = j%config.board_w;
					CV_MAT_ELEM(*object_points_right,float,i,2) = 0.0f;
				}
				CV_MAT_ELEM(*point_counts_right, int,successes,0) = board_n;
				successes++;
			}else{
				printf("Chessboard not found\n");

			}
		}

		//_stitchedImage = cvCreateImage(cvSize(frameLeftSize.width + frameRightSize.width ,frameLeftSize.height), IPL_DEPTH_8U, 1); 
		_stitchedImage = cvCreateImage(cvSize(640 ,240), IPL_DEPTH_8U, 3); 
		cvSetImageROI(_stitchedImage, cvRect(0, 0, 320, 240)); 
		cvCopy(frame_left, _stitchedImage); 
		cvSetImageROI(_stitchedImage, cvRect(320, 0, 320, 240)); 
		cvCopy(frame_right, _stitchedImage); 
		cvResetImageROI(_stitchedImage); 
		cvShowImage("Stitched Window", _stitchedImage); 


		int c = cvWaitKey(15);

		if(takeScreenshot==true){
			Sleep(1000);
			takeScreenshot=false;
		}
		
		if(c == 'n'){
			takeScreenshot=true;
        }

        //Handle pause/unpause and ESC
        if(c == 'p'){
            printf("\nPause\n");
			c = 0;
            while(c != 'p' && c != 27){
                c = cvWaitKey(250);
            }
			printf("Restart\n");
        }

        if(c == 27) {
			printf("Exit\n");	
			return; 
		}

        frame_left = cvQueryFrame( cameraLeft );
		frame_right = cvQueryFrame( cameraRight );

    } 

	/*
	printf("Running normal calibration ...");
	computeCalibration(successes,board_n,image_points_left,object_points_left,point_counts_left, frameLeftSize,"left");
	computeCalibration(successes,board_n,image_points_right,object_points_right,point_counts_right, frameRightSize,"right");

	
	CvMat *intrinsic_left = (CvMat*)cvLoad("Intrinsics_left.xml");
    CvMat *distortion_left = (CvMat*)cvLoad("Distortion_left.xml");
	CvMat *intrinsic_right = (CvMat*)cvLoad("Intrinsics_right.xml");
    CvMat *distortion_right = (CvMat*)cvLoad("Distortion_right.xml");
	*/

	computeStereoCalibration(image_points_left,object_points_left,point_counts_left,frameLeftSize,image_points_right,object_points_right,point_counts_right,frameRightSize);
}

void computeSingleCalibration(int nbrImage,int board_n,CvMat* image_points,CvMat* object_points,CvMat* point_counts, CvSize imgSize,char *filePrefix)
{
    //ALLOCATE MATRICES ACCORDING TO HOW MANY CHESSBOARDS FOUND
    CvMat* object_points2 = cvCreateMat(nbrImage*board_n,3,CV_32FC1);
    CvMat* image_points2 = cvCreateMat(nbrImage*board_n,2,CV_32FC1);
    CvMat* point_counts2 = cvCreateMat(nbrImage,1,CV_32SC1);
	CvMat* intrinsic_matrix = cvCreateMat(3,3,CV_32FC1);
    CvMat* distortion_coeffs = cvCreateMat(5,1,CV_32FC1);

	for(int i = 0; i<nbrImage*board_n; ++i) {
        CV_MAT_ELEM( *image_points2, float, i, 0) =  CV_MAT_ELEM( *image_points, float, i, 0);
        CV_MAT_ELEM( *image_points2, float,i,1) =  CV_MAT_ELEM( *image_points, float, i, 1);
        CV_MAT_ELEM(*object_points2, float, i, 0) =  CV_MAT_ELEM( *object_points, float, i, 0) ;
        CV_MAT_ELEM( *object_points2, float, i, 1) =  CV_MAT_ELEM( *object_points, float, i, 1) ;
        CV_MAT_ELEM( *object_points2, float, i, 2) =  CV_MAT_ELEM( *object_points, float, i, 2) ;
    }
    for(int i=0; i<nbrImage; ++i){ //These are all the same number
        CV_MAT_ELEM( *point_counts2, int, i, 0) =   CV_MAT_ELEM( *point_counts, int, i, 0);
    }

	// At this point we have all of the chessboard corners we need.
    // Initialize the intrinsic matrix such that the two focal
    // lengths have a ratio of 1.0
    //
    CV_MAT_ELEM( *intrinsic_matrix, float, 0, 0 ) = 1.0f;
    CV_MAT_ELEM( *intrinsic_matrix, float, 1, 1 ) = 1.0f;
    //CALIBRATE THE CAMERA!
    cvCalibrateCamera2( object_points2, image_points2,
                        point_counts2, imgSize,
                        intrinsic_matrix, distortion_coeffs,
                        NULL, NULL,0 //CV_CALIB_FIX_ASPECT_RATIO
                        );
    // SAVE THE INTRINSICS AND DISTORTIONS
	char fileName[40]; 
	sprintf(fileName,"Intrinsics_%s.xml", filePrefix); 
    cvSave(fileName,intrinsic_matrix);
	sprintf(fileName,"Distortion_%s.xml", filePrefix); 
    cvSave(fileName,distortion_coeffs);
}

void computeStereoCalibration(
	CvMat* image_points_left,CvMat* object_points_left,CvMat* point_counts_left, CvSize frameLeftSize,
	CvMat* image_points_right,CvMat* object_points_right,CvMat* point_counts_right,CvSize frameRightSize){

	/** -------------------------------------------------------
	* STEREO CALIBRATION
	* ------------------------------------------------------- */
	printf("Running stereo calibration ...");
	 // ARRAY AND VECTOR STORAGE:
    double M1[3][3], M2[3][3], D1[5], D2[5];
    double R[3][3], T[3], E[3][3], F[3][3];
    CvMat _M1 = cvMat(3, 3, CV_64F, M1 );
    CvMat _M2 = cvMat(3, 3, CV_64F, M2 );
    CvMat _D1 = cvMat(1, 5, CV_64F, D1 );
    CvMat _D2 = cvMat(1, 5, CV_64F, D2 );
    CvMat _R = cvMat(3, 3, CV_64F, R );
    CvMat _T = cvMat(3, 1, CV_64F, T );
    CvMat _E = cvMat(3, 3, CV_64F, E );
    CvMat _F = cvMat(3, 3, CV_64F, F );


	
	//CALIBRATE THE STEREO CAMERAS
    cvStereoCalibrate( object_points_left, image_points_left,
        image_points_right,point_counts_left, &_M1, &_D1, &_M2, &_D2,frameLeftSize, &_R, &_T, &_E, &_F,
        cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
        CV_CALIB_FIX_ASPECT_RATIO + CV_CALIB_ZERO_TANGENT_DIST + CV_CALIB_SAME_FOCAL_LENGTH
    );
	printf(" done\n");

	cvSave("M1.xml",&_M1);
    cvSave("D1.xml",&_D1);
    cvSave("M2.xml",&_M2);
    cvSave("D2.xml",&_D2);

	cvSave("R.xml",&_R);
    cvSave("T.xml",&_T);
    cvSave("E.xml",&_E);
    cvSave("F.xml",&_F);

	/** -------------------------------------------------------
	* STEREO RECTIFICATION
	* ------------------------------------------------------- */
	printf("Running stereo rectification ...");

	double R1[3][3], R2[3][3], P1[3][4], P2[3][4];
	double Q[4][4];
    CvMat _R1 = cvMat(3, 3, CV_64F, R1);
    CvMat _R2 = cvMat(3, 3, CV_64F, R2);
	CvMat _P1 = cvMat(3, 4, CV_64F, P1);
    CvMat _P2 = cvMat(3, 4, CV_64F, P2);
	CvMat _Q = cvMat(4,4, CV_64F, Q);

    cvStereoRectify( &_M1, &_M2, &_D1, &_D2, frameLeftSize,&_R, &_T,&_R1, &_R2, &_P1, &_P2, &_Q,0/*CV_CALIB_ZERO_DISPARITY*/ );
	

	cvSave("R1.xml",&_R1);
    cvSave("P1.xml",&_P1);
	cvSave("R2.xml",&_R2);
    cvSave("P2.xml",&_P2);
	cvSave("Q.xml",&_Q);
    

	printf(" done\n");

}

distortionMapsStruct computeDistortionMapping(CvSize frameLeftSize, CvSize frameRightSize){
	printf("Compute distortion mapping ...");
	printf("Read Calibration parameters from xml files : R1,P1,R2,P2,Q");
	
	
	//Retrieve Calibration parameters
	CvMat *_M1 = (CvMat*)cvLoad("M1.xml");
    CvMat *_D1 = (CvMat*)cvLoad("D1.xml");
	CvMat *_M2 = (CvMat*)cvLoad("M2.xml");
    CvMat *_D2 = (CvMat*)cvLoad("D2.xml");

	CvMat *_R1 = (CvMat*)cvLoad("R1.xml");
    CvMat *_P1 = (CvMat*)cvLoad("P1.xml");
	CvMat *_R2 = (CvMat*)cvLoad("R2.xml");
    CvMat *_P2 = (CvMat*)cvLoad("P2.xml");
	CvMat *_Q = (CvMat*)cvLoad("Q.xml");

	/** -------------------------------------------------------
	* COMPUTE DISTORTION MAPPING
	* ------------------------------------------------------- */	
	
	CvMat* mx1 = cvCreateMat( frameLeftSize.height,frameLeftSize.width, CV_32F );
    CvMat* my1 = cvCreateMat( frameLeftSize.height,frameLeftSize.width, CV_32F );
    CvMat* mx2 = cvCreateMat( frameRightSize.height,frameRightSize.width, CV_32F );
    CvMat* my2 = cvCreateMat( frameRightSize.height,frameRightSize.width, CV_32F );

	cvInitUndistortRectifyMap(_M1,_D1,_R1,_P1,mx1,my1);
    cvInitUndistortRectifyMap(_M2,_D2,_R2,_P2,mx2,my2);

	/*cvSave("mx1.xml",mx1);
    cvSave("my1.xml",my1);
    cvSave("mx2.xml",mx2);
    cvSave("my2.xml",my2);*/
	printf(" done\n");

	distortionMapsStruct retMaps;
	retMaps.mx1 = mx1;
	retMaps.mx2 = mx2;
	retMaps.my1 = my1;
	retMaps.my2 = my2;

	return retMaps;
}

void startStereoMatching(CvCapture* cameraLeft,CvCapture* cameraRight){

    IplImage* frame_left = cvQueryFrame( cameraLeft );
    IplImage* frame_right = cvQueryFrame( cameraRight );
	CvSize frameLeftSize = cvGetSize(frame_left);
	CvSize frameRightSize = cvGetSize(frame_right);

	IplImage* frame_left_rectified = cvCreateImage(frameLeftSize, IPL_DEPTH_8U, 3); //cvCreateMat( frameLeftSize.height,frameLeftSize.width, CV_8U );
    IplImage* frame_right_rectified = cvCreateImage(frameRightSize, IPL_DEPTH_8U, 3); 
	IplImage *frame_left_gray = cvCreateImage(frameLeftSize,IPL_DEPTH_8U,1);
	IplImage *frame_right_gray = cvCreateImage(frameRightSize,IPL_DEPTH_8U,1);
	IplImage *frame_left_gray_rectified = cvCreateImage(frameLeftSize,IPL_DEPTH_8U,1);
	IplImage *frame_right_gray_rectified = cvCreateImage(frameRightSize,IPL_DEPTH_8U,1);
    IplImage* _stitchedImage = cvCreateImage(cvSize(320*2 ,240*3), IPL_DEPTH_8U, 3);

    CvMat* disparity_left = cvCreateMat( frameLeftSize.height,frameLeftSize.width, CV_16S );
	CvMat* vdisp = cvCreateMat( frameLeftSize.height,frameLeftSize.width, CV_8U ); //Visible disparity
    IplImage* dispColor = cvCreateImage(frameRightSize, IPL_DEPTH_8U, 3);

	distortionMapsStruct distortionMaps = computeDistortionMapping(frameLeftSize,frameRightSize);
	CvMat* mx1 = distortionMaps.mx1;
	CvMat* mx2 = distortionMaps.mx2;
	CvMat* my1 = distortionMaps.my1;
	CvMat* my2 = distortionMaps.my2;


    /*CvStereoBMState *BMState = cvCreateStereoBMState();
        assert(BMState != 0);
        BMState->preFilterSize=41;
        BMState->preFilterCap=31;
        BMState->SADWindowSize=41;
        BMState->minDisparity=-64;
        BMState->numberOfDisparities=128;
        BMState->textureThreshold=10;
        BMState->uniquenessRatio=15;
*/


    //Daisy descriptor configuration
    int16_t rad, radq, thq, histq;
    rad=15;
    radq=4;
    thq=8;
    histq=8;

    int16_t dispMin=0;
    int16_t dispMax=80;
    int16_t width,height;
    width= (int16_t) frame_left_gray_rectified->width;
    height=(int16_t) frame_left_gray_rectified->height;

    daisy* desc_left = new daisy();
    desc_left->verbose( 0 ); // 0,1,2,3 -> how much output do you want while running
    //desc_left->disable_interpolation();
    desc_left->set_image(frame_left->imageData,frameLeftSize.height,frameLeftSize.width);
    desc_left->set_parameters(rad, radq, thq, histq); // default values are 15,3,8,8
    long int wszLeft =desc_left->compute_workspace_memory();
    long int dszLeft =desc_left->compute_descriptor_memory();
    float *descriptorLeft = new float[dszLeft];
    float *workspaceLeft = new float[wszLeft];
    desc_left->set_descriptor_memory(descriptorLeft,dszLeft);
    desc_left->set_workspace_memory( workspaceLeft, wszLeft );

    daisy* desc_right = new daisy();
    desc_right->verbose( 0 ); // 0,1,2,3 -> how much output do you want while running
    //desc_right->disable_interpolation();
    desc_right->set_image(frame_right->imageData,frameRightSize.height,frameRightSize.width);
    desc_right->set_parameters(rad, radq, thq, histq); // default values are 15,3,8,8
    long int wszRight=desc_right->compute_workspace_memory();
    long int dszRight =desc_right->compute_descriptor_memory();
    float *descriptorRight = new float[dszRight];
    float *workspaceRight = new float[wszRight];
    desc_right->set_descriptor_memory(descriptorRight,dszRight);
    desc_right->set_workspace_memory( workspaceRight, wszRight );

    cvNamedWindow("Stitched Window - Rectify", CV_WINDOW_AUTOSIZE);
	while(true) {
        try
        {

            printf(".");

            cvRemap( frame_left, frame_left_rectified, mx1, my1 );
            cvRemap( frame_right, frame_right_rectified, mx2, my2 );

            cvCvtColor(frame_left, frame_left_gray, CV_BGR2GRAY);
            cvCvtColor(frame_right, frame_right_gray, CV_BGR2GRAY);

            cvRemap( frame_left_gray, frame_left_gray_rectified, mx1, my1 );
            cvRemap( frame_right_gray, frame_right_gray_rectified, mx2, my2 );


            /** OPENCV STEREO MATCHING
            cvFindStereoCorrespondenceBM( frame_left_gray_rectified, frame_right_gray_rectified, disparity_left, BMState);
            cvNormalize( disparity_left, vdisp, 0, 256, CV_MINMAX );
            cvShowImage( "disparity", vdisp );
            */

            /** DAISY MATCHING */


            desc_left->set_image(frame_left_gray_rectified->imageData,height,width);
            desc_left->initialize_single_descriptor_mode();
            desc_left->compute_descriptors(); // precompute all the descriptors (NOT NORMALIZED!)
            desc_left->normalize_descriptors();


            desc_right->set_image(frame_right_gray_rectified->imageData,height,width);
            desc_right->initialize_single_descriptor_mode();
            desc_right->compute_descriptors(); // precompute all the descriptors (NOT NORMALIZED!)
            desc_right->normalize_descriptors();

            float* dzyL =desc_left->get_dense_descriptors();
            float* dzyR =desc_right->get_dense_descriptors();
            parallel_for(blocked_range<size_t>(1,height,160),ComputeLeftDisparity(dzyL,dzyR,(int16_t *)disparity_left->data.ptr,width,height,dispMin,dispMax,rad,radq,thq,histq));
            desc_left->reset();
            desc_right->reset();

            cvNormalize( disparity_left, vdisp, 0, 255, CV_MINMAX );

            //SYSTEMTIME timeBefore,timeAfter;
           // GetSystemTime(&timeBefore);
            for(int i=0;i<frameRightSize.height*frameRightSize.width;i++){
                uint8_t valueDisp = (uint8_t) vdisp->data.ptr[i];
                dispColor->imageData[3*i] = 255-valueDisp; //B
                dispColor->imageData[3*i+1] = 0; //G
                dispColor->imageData[3*i+2] = valueDisp; //R
            }
           // GetSystemTime(&timeAfter);
            //WORD millis =((timeAfter.wSecond * 1000) + timeAfter.wMilliseconds) -((timeBefore.wSecond * 1000) + timeBefore.wMilliseconds);
            //printf( "%ld\n", millis );


            cvSetImageROI(_stitchedImage, cvRect(0, 0, 320, 240));
            cvCopy(frame_left, _stitchedImage);
            cvSetImageROI(_stitchedImage, cvRect(320, 0, 320, 240));
            cvCopy(frame_right, _stitchedImage);
            cvResetImageROI(_stitchedImage);
            cvSetImageROI(_stitchedImage, cvRect(0, 240, 320, 240));
            cvCopy(frame_left_rectified, _stitchedImage);
            cvResetImageROI(_stitchedImage);
            cvSetImageROI(_stitchedImage, cvRect(320, 240, 320, 240));
            cvCopy(frame_right_rectified, _stitchedImage);
            cvSetImageROI(_stitchedImage, cvRect(0, 2*240, 320, 240));
            cvCopy(dispColor, _stitchedImage);
            cvResetImageROI(_stitchedImage);
            cvShowImage("Stitched Window - Rectify", _stitchedImage);


            int c = cvWaitKey(15);
            //Handle pause/unpause and ESC

            if(c == 'p'){
                c = 0;
                while(c != 'p' && c != 27){
                    c = cvWaitKey(250);
                }
            }

            if(c == 27)
            {
                cvDestroyAllWindows();
                return;
            }

            frame_left = cvQueryFrame( cameraLeft );
            frame_right = cvQueryFrame( cameraRight );
        }catch( cv::Exception& e ){
            const char* err_msg = e.what();
            printf(err_msg);
            return;
        }catch(std::exception const& e){
            std::cout << "Error: " << e.what() <<std::endl;
            return;
        }
    }
}
