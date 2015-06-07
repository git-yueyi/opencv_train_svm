#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>  
#include <direct.h>//_mkdir()  
#include "opencv2\opencv.hpp"  
#include <ml.h>

using namespace std;
using namespace cv;

cv::Directory dir;

string path = "E:/imgdata"; 

vector<string> folder_names;

vector<string> img_names;

#define Exten_IMG "*.jpg"

int getNumImg(string path_){
	
	string img ="*";

	vector<string> allimg_names = dir.GetListFilesR(path_,img,true);
	
	return allimg_names.size();
}

void getFoldersName(string path_,vector<string> & folder_names_){
	
	string doc = "*";

	bool addPath = false;

	folder_names_ = dir.GetListFolders(path_,doc,addPath); 
}

void getFileName(string path_,vector<string> & file_name_){
	
	string img = Exten_IMG;

	bool addPath = false;

	file_name_ = dir.GetListFiles(path_,img,addPath);
}

void test_svm(string svmxml){
	CvSVM svm;

	svm.load(svmxml.c_str());
	
	int NumImgs =  getNumImg(path);

	Mat imgFeature = Mat(1,3787,CV_32FC1,Scalar::all(0));

	folder_names.clear();

	getFoldersName(path,folder_names);

	int n_imgs = 0;

	for (int fidx = 0;fidx<folder_names.size();fidx++)
	{
		img_names.clear();

		getFileName(path+"/"+folder_names[fidx],img_names);

		float err = 0.0;

		for (int imgidx = 18 ;imgidx<img_names.size();imgidx++)
		{
			Mat img = imread(path+"/"+ folder_names[fidx] +"/"+ img_names[imgidx],0);

			resize(img,img,Size(64,128));

			threshold(img,img,125,255,CV_THRESH_BINARY);

			HOGDescriptor *hog=new HOGDescriptor(cvSize(64,128),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);

			vector<float> descriptors;

			hog->compute(img,descriptors,Size(1,1),Size(0,0));

			for (int desidx = 0;desidx<descriptors.size();desidx++)
			{
				imgFeature.at<float>(0,desidx) = descriptors[desidx];
			}

			cv::Moments moments = cv::moments(img);

			double hu_moments[7];

			cv::HuMoments(moments,hu_moments);

			for (int huidx = 0;huidx<7;huidx++)
			{
				imgFeature.at<float>(0,3780+huidx) = hu_moments[huidx];
			}
			hog->~HOGDescriptor();

			descriptors.clear();

			img.~Mat();

			
			if(fidx!=svm.predict(imgFeature)){
				err  += 1.0;
			}
		}
		//cout<<"当前文件路劲: "<<path+"/"+folder_names[fidx]<<endl;
		err = err/(float)(img_names.size()-18);
		cout<<folder_names[fidx]<<" 错误率："<<err<<endl;
	}
	
}

void cal_HogHu(Mat & allHogFeatureMat,Mat & labelMat){

	int NumImgs =  getNumImg(path);

	allHogFeatureMat = Mat(NumImgs,3787,CV_32FC1,Scalar::all(1));
	
	labelMat = Mat(NumImgs,1,CV_32FC1,Scalar::all(0));

	folder_names.clear();

	getFoldersName(path,folder_names);

	int n_imgs = 0;

	for (int fidx=0;fidx<folder_names.size();fidx++)
	{
		cout<<"当前文件路劲: "<<path+"/"+folder_names[fidx]<<endl;

		img_names.clear();

		getFileName(path+"/"+folder_names[fidx],img_names);

		for (int imgidx=0;imgidx<img_names.size()&&imgidx<18;imgidx++)
		{
			cout<<path+"/"+ folder_names[fidx] +"/" + img_names[imgidx]<<endl;

			Mat img = imread(path+"/"+ folder_names[fidx] +"/"+ img_names[imgidx],0);

			resize(img,img,Size(64,128));
			
			threshold(img,img,125,255,CV_THRESH_BINARY);

			cout<<"图片通道："<<img.channels()<<" w:"<<img.cols<<" h: "<<img.rows<<" n: "<<n_imgs<<" label: "<<fidx<<endl;

			imshow("img",img);

			waitKey(20);

			HOGDescriptor *hog=new HOGDescriptor(cvSize(64,128),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);

			vector<float> descriptors;

			hog->compute(img,descriptors,Size(1,1),Size(0,0));
			
			labelMat.at<float>(n_imgs,0) = fidx;

			for (int desidx = 0;desidx<descriptors.size();desidx++)
			{
				allHogFeatureMat.at<float>(n_imgs,desidx) = descriptors[desidx];
			}

			cv::Moments moments = cv::moments(img);

			double hu_moments[7];

			cv::HuMoments(moments,hu_moments);

			for (int huidx = 0;huidx<7;huidx++)
			{
				allHogFeatureMat.at<float>(n_imgs,3780+huidx) = hu_moments[huidx];
			}
			hog->~HOGDescriptor();

			descriptors.clear();

			n_imgs++;

			img.~Mat();



		}
	}




}

void trainsvm(string svmname){

	Mat TrainSet;

	Mat labelMat;

	cal_HogHu(TrainSet,labelMat);

	CvSVM svm ;//新建一个SVM      

	CvSVMParams param;//这里是SVM训练相关参数 

	CvTermCriteria criteria;    

	criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );  

	param = CvSVMParams( CvSVM::C_SVC, CvSVM::LINEAR, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );

	svm.train( TrainSet, labelMat, Mat(), Mat(), param );//训练数据     

	svm.save(svmname.c_str());
}

int main(void){

	namedWindow("img");

	//trainsvm("svm_hoghu_char_liner18.xml");

	test_svm("svm_hoghu_char_liner18.xml");

	waitKey(0);

	//trainsvm();

	//CvSVM svm;

	//svm.load("svm_hoghu_char_LINER.xml");

	//double  t=(double)cvGetTickCount();

	//Mat img = imread("F:\\visual studio 2010\\Projects\\GBpic_rec\\GBpic_rec\\gb\\gbB.bmp",0);

	//Mat testData(1,3787,CV_32FC1,Scalar::all(0));

	//resize(img,img,Size(64,128));

	//threshold(img,img,125,255,CV_THRESH_BINARY);

	//imshow("img",img);
	
	//waitKey(200);

	//HOGDescriptor *hog=new HOGDescriptor(cvSize(64,128),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);

	//vector<float> descriptors;

	//hog->compute(img,descriptors,Size(1,1),Size(0,0));

	//for (int hogidx=0;hogidx<descriptors.size();hogidx++)
	//{
	//	testData.at<float>(0,hogidx) = descriptors[hogidx];
	//}

	//cv::Moments moments = cv::moments(img);

	//double hu_moments[7];

	//cv::HuMoments(moments,hu_moments);

	//for (int huidx=0;huidx<7;huidx++)
	//{
	//	testData.at<float>(0,3780+ huidx) = hu_moments[huidx];
	//}

	//int ret = svm.predict(testData);

	//t=((double)cvGetTickCount() - t)/(cvGetTickFrequency()*1000);
	//
	//cout<<"the total time: "<<t<<" ms"<<endl;

	//cout<<ret<<endl;

	//waitKey(0);

	return 0;
}