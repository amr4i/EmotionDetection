/*
* Copyright (c) 2016 Nrupatunga
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*
*/
#include <numeric>
#include "gist.hpp"

#if 0
#pragma comment(lib, "opencv_core2413d.lib") // core functionalities
#pragma comment(lib, "opencv_highgui2413d.lib") //GUI
#pragma comment(lib, "opencv_imgproc2413d.lib") // Histograms, Edge detection
#else
#pragma comment(lib, "opencv_core2413.lib") // core functionalities
#pragma comment(lib, "opencv_highgui2413.lib") //GUI
#pragma comment(lib, "opencv_imgproc2413.lib") // Histograms, Edge detection
#endif

Gist::Gist ( )
{
	return ;
}		/* -----  end of method Gist::Gist  ----- */

Gist::Gist(String strFolderPath, String strFileNames, String strGistFile)
{
	strImgFolderPath = strFolderPath;
	strGistFilePath  = strGistFile;
	strFileNamesPath = strFileNames;
	s32GistMode      = 1;

	//Populate all the images
	glob(strFolderPath, vecStrImgPath);

}

Gist::GistDescriptor * Gist::compute(Params * pObjParams)
{
	objParams.s32NumBlocks      = pObjParams->s32NumBlocks;
	objParams.vecIntOrientation = pObjParams->vecIntOrientation;
	objParams.s32NumPad         = 32;                            // Number of pixels to pad around the image
	objParams.s32PreFilt        = 4;
	objParams.s32NumBlocks      = 4;

	//Create Gabor filters
	createGabor(GIST_WIDTH_HEIGHT, GIST_WIDTH_HEIGHT);
	//Calculate the feature length
	int s32FeatureLen = objParams.s32NumBlocks * objParams.s32NumBlocks * vecGaborFilters.size();
	float *pfGistFeature = new float[512];

	return &objGistDescriptor;
}

Gist::GistDescriptor* Gist::compute(Mat sMatInput, Params *pObjParams)
{
	objParams.s32NumBlocks      = pObjParams->s32NumBlocks;
	objParams.vecIntOrientation = pObjParams->vecIntOrientation;
	objParams.s32NumPad         = 32;                            // Number of pixels to pad around the image
	objParams.s32PreFilt        = 4;
	objParams.s32NumBlocks      = 4;

	Mat sMatInputGray;
	if (sMatInput.channels() == 3) {
		cvtColor(sMatInput, sMatInputGray, COLOR_BGR2GRAY);
	} else if (sMatInput.channels() == 3) {
		cvtColor(sMatInput, sMatInputGray, COLOR_BGRA2GRAY);
	} else {
		sMatInputGray = sMatInput;
	}
	if (sMatInputGray.rows != GIST_WIDTH_HEIGHT || sMatInputGray.cols != GIST_WIDTH_HEIGHT) {
		resize(sMatInputGray, sMatInputGray, Size(GIST_WIDTH_HEIGHT, GIST_WIDTH_HEIGHT), INTER_NEAREST);
	}
	

	//Create Gabor filters
	createGabor(GIST_WIDTH_HEIGHT, GIST_WIDTH_HEIGHT);
	//Calculate the feature length
	int s32FeatureLen = objParams.s32NumBlocks * objParams.s32NumBlocks * vecGaborFilters.size();
	float *pfGistFeature = new float[s32FeatureLen];
	objGistDescriptor.pfGistDescriptor = pfGistFeature;

	vector<float> vecGistFeature;
	normalize(sMatInputGray, sMatInputGray, 0, 255, NORM_MINMAX);

	Mat sMatPrefilt = Mat(Size(sMatInputGray.cols, sMatInputGray.rows), CV_32FC1);
	prefilter(sMatInputGray, sMatPrefilt);

	Mat sMatPrefiltPad;
	copyMakeBorder(sMatPrefilt, sMatPrefiltPad, objParams.s32NumPad, objParams.s32NumPad, objParams.s32NumPad, objParams.s32NumPad, BORDER_REFLECT);
	Mat sMatPrefiltfft;
	dft(sMatPrefiltPad, sMatPrefiltfft, DFT_COMPLEX_OUTPUT);
	sMatPrefiltPad.release();


	//Caution: sMatFiltOut should be integer multiple of block size, 
	int s32StepW = floor(sMatInputGray.cols / (1.0*objParams.s32NumBlocks));
	int s32StepH = floor(sMatInputGray.rows / (1.0*objParams.s32NumBlocks));

	for (int i = 0; i < vecGaborFilters.size(); i++) {
		Mat sMatGabor; Mat sMatPlanes[2];
		Mat sMatPrefiltifft; Mat sMatFiltOutTwo;
		Mat sMatBlkOut, sMatFiltOutOne;

		sMatGabor = vecGaborFilters.at(i);
		split(sMatPrefiltfft, sMatPlanes);
		sMatPlanes[0] = sMatPlanes[0].mul(sMatGabor);
		sMatPlanes[1] = sMatPlanes[1].mul(sMatGabor);
		merge(sMatPlanes, 2, sMatFiltOutOne);
		idft(sMatFiltOutOne, sMatPrefiltifft, DFT_SCALE);
		split(sMatPrefiltifft, sMatPlanes);
		magnitude(sMatPlanes[0], sMatPlanes[1], sMatPlanes[0]);

		Rect sRectROI(objParams.s32NumPad, objParams.s32NumPad, sMatInputGray.cols, sMatInputGray.rows);
		sMatPlanes[0](sRectROI).copyTo(sMatFiltOutTwo);
		//Caution: sMatFiltOut should be integer multiple of block size, 
		for (int i = 0; i < sMatInputGray.rows; i = i + s32StepH) {
			for (int j = 0; j < sMatInputGray.cols; j = j + s32StepW) {
				int s32ImgW = (i + s32StepW) < sMatInputGray.cols ? s32StepW : (sMatInputGray.cols - i);
				int s32ImgH = (i + s32StepH) < sMatInputGray.rows ? s32StepH : (sMatInputGray.rows- j);
				sRectROI = Rect(i, j, s32StepW, s32StepH);
				sMatBlkOut = sMatFiltOutTwo(sRectROI);
				Scalar scMean = mean(sMatBlkOut);
				vecGistFeature.push_back(scMean[0]);
			}
		}
	}
	objGistDescriptor.s32Length = vecGistFeature.size();
	for (int i = 0; i < vecGistFeature.size(); i++) {
		objGistDescriptor.pfGistDescriptor[i] = vecGistFeature.at(i);
	}
	return &objGistDescriptor;
}

void Gist::createGabor(int s32ImgWidth, int s32ImgHeight)
{
	int s32NumScales  = objParams.vecIntOrientation.size();
	int s32NumFilters = accumulate(objParams.vecIntOrientation.begin(), objParams.vecIntOrientation.end(), 0);

	vector< vector<float> > vec2DfloatFiltParams(s32NumFilters, vector<float>(s32NumScales, 0));
	
	int s32Index = 0;
	for (int i = 0; i < s32NumScales; i++) {
		for (int j = 0; j < objParams.vecIntOrientation.at(i); j++) {
			vec2DfloatFiltParams[s32Index].at(0) = (float)0.35;
			vec2DfloatFiltParams[s32Index].at(1) = (float)0.3/(pow(1.85, i));
			vec2DfloatFiltParams[s32Index].at(2) = (float)(16*objParams.vecIntOrientation.at(i)*objParams.vecIntOrientation.at(i))/(32.0*32.0);
			vec2DfloatFiltParams[s32Index].at(3) = (float)CV_PI*j/(objParams.vecIntOrientation.at(i));
			s32Index++;
		}
	}

	int s32ImgWPad = s32ImgWidth  + 2 * objParams.s32NumPad;
	int s32ImgHPad = s32ImgHeight + 2 * objParams.s32NumPad;
	objParams.s32ImgWHPad = s32ImgWPad;
	
	Mat sMatFx        = Mat(Size(s32ImgWPad, s32ImgHPad), CV_32SC1);
	Mat sMatFy        = Mat(Size(s32ImgWPad, s32ImgHPad), CV_32SC1);
	int s32ImgWPadby2 = s32ImgWPad >> 1;
	for (int i = 0; i < sMatFx.cols; i++) {
		sMatFx.at<int>(0, i) = -s32ImgWPadby2--;
	}

	for (int i = 0; i < sMatFx.rows; i++) {
		sMatFx.row(0).copyTo(sMatFx.row(i));
	}
	transpose(sMatFx, sMatFy);
	//vecMatCmplx.at(0) = sMatFx;
	//vecMatCmplx.at(1) = sMatFy;
	//Mat sMatFCmplx;
	//merge(vecMatCmplx, sMatFCmplx);
	//sMatFCmplx.convertTo(sMatFCmplx, CV_32FC2);

	sMatFx.convertTo(sMatFx, CV_32FC1);
	sMatFy.convertTo(sMatFy, CV_32FC1);
	Mat sMatAngle = Mat(Size(sMatFx.cols, sMatFx.rows), CV_32FC1);
	for (int i = 0; i < sMatFx.rows; i++) {
		float *pfDataFx    = sMatFx.ptr<float>(i);
		float *pfDataFy    = sMatFy.ptr<float>(i);
		float *pfDataAngle = sMatAngle.ptr<float>(i);
		for (int j = 0; j < sMatFx.cols; j++) {
			float fAngle = fastAtan2(pfDataFy[j], pfDataFx[j]);
			fAngle = (fAngle > 180) ? (fAngle - 360) : fAngle;
			pfDataAngle[j] = fAngle*CV_PI/180.0;
		}
	}
	Mat sMatfftshiftAngle = Mat::zeros(sMatAngle.size(), CV_32FC1);
	fftshift(sMatAngle, sMatfftshiftAngle, floor(s32ImgWPad / 2.0), floor(s32ImgHPad / 2.0));

	Mat sMatFxTemp = sMatFx.mul(sMatFx);
	Mat sMatFyTemp = sMatFy.mul(sMatFy);

	sMatFxTemp.convertTo(sMatFxTemp, CV_32FC1);
	sMatFyTemp.convertTo(sMatFyTemp, CV_32FC1);

	Mat sMatF = sMatFxTemp + sMatFyTemp;
	sqrt(sMatF, sMatF);

	Mat sMatfftshiftF = Mat::zeros(sMatF.size(), CV_32FC1);
	fftshift(sMatF, sMatfftshiftF, floor(s32ImgWPad/2.0), floor(s32ImgHPad/2.0));

	for (int i = 0; i < s32NumFilters; i++) {
		Mat sMatfftshiftAngleTemp = sMatfftshiftAngle + vec2DfloatFiltParams[i].at(3);
		Mat sMatIndexOne, sMatIndexTwo;
		Mat sMatGaborFilter;

		sMatIndexOne = (sMatfftshiftAngleTemp < (-CV_PI));
		sMatIndexTwo = (sMatfftshiftAngleTemp > (CV_PI));
		normalize(sMatIndexOne, sMatIndexOne, 0, 1, CV_MINMAX);
		normalize(sMatIndexTwo, sMatIndexTwo, 0, 1, CV_MINMAX);
		sMatIndexOne.convertTo(sMatIndexOne, CV_32FC1);
		sMatIndexTwo.convertTo(sMatIndexTwo, CV_32FC1);

		sMatIndexOne = sMatIndexOne * 2 * CV_PI;
		sMatIndexTwo = sMatIndexTwo * 2 * CV_PI;

		sMatfftshiftAngleTemp = sMatfftshiftAngleTemp + sMatIndexOne - sMatIndexTwo;

		Mat sMatExpTermOne = (sMatfftshiftF/(objParams.s32ImgWHPad*vec2DfloatFiltParams[i].at(1)) - 1);
		sMatExpTermOne     = sMatExpTermOne.mul(sMatExpTermOne);
		sMatExpTermOne     = -10 * vec2DfloatFiltParams[i].at(0) * sMatExpTermOne;
		Mat sMatExpTermTwo = sMatfftshiftAngleTemp.mul(sMatfftshiftAngleTemp);
		sMatExpTermTwo     = 2 * vec2DfloatFiltParams[i].at(2) * CV_PI * sMatExpTermTwo;
		Mat sMatExpTerm    = sMatExpTermOne - sMatExpTermTwo;
		exp(sMatExpTerm, sMatGaborFilter);
		vecGaborFilters.push_back(sMatGaborFilter);
	}

	
#if DEBUG
	s32Index = 0;
	for (int i = 0; i < s32NumScales; i++) {
		for (int j = 0; j < objParams.vecIntOrientation.at(i); j++) {
			cout << vec2DfloatFiltParams[s32Index].at(0) << " ";
			cout << vec2DfloatFiltParams[s32Index].at(1) << " ";
			cout << vec2DfloatFiltParams[s32Index].at(2) << " ";
			cout << vec2DfloatFiltParams[s32Index].at(3) << " " << endl;
			s32Index++;
		}
	}
#endif
}

void Gist::prefilter(Mat& sMatInput, Mat& sMatOutput)
{
	float fFreq  = 4;// 4cycles/image
	int   s32Pad = 5;
	float fS1    = fFreq / sqrt(log(2));

#if 0
		float *pu8Data = new float [65536];
	{
		FILE *fp = fopen("img.raw", "r");
		fread(pu8Data, 256 * 256, 4, fp);
		fclose(fp);
	}
	Mat sMatInput = Mat(Size(256, 256), CV_32FC1, pu8Data, Mat::AUTO_STEP);
	transpose(sMatInput, sMatInput);
#endif

	Mat sMatInputTemp = sMatInput + 1;
	Mat sMatOutputTemp;
	sMatInputTemp.convertTo(sMatOutputTemp, CV_32FC1);
	sMatInputTemp.release();
	log(sMatOutputTemp, sMatOutputTemp);
	copyMakeBorder(sMatOutputTemp, sMatOutputTemp, s32Pad, s32Pad, s32Pad, s32Pad, BORDER_REFLECT);

	Mat sMatFx        = Mat(Size(sMatOutputTemp.cols, sMatOutputTemp.rows), CV_32SC1);
	Mat sMatFy        = Mat(Size(sMatOutputTemp.cols, sMatOutputTemp.rows), CV_32SC1);
	int s32ImgWPadby2 = sMatOutputTemp.cols >> 1;
	for (int i = 0; i < sMatFx.cols; i++) {
		sMatFx.at<int>(0, i) = -s32ImgWPadby2--;
	}

	for (int i = 0; i < sMatFx.rows; i++) {
		sMatFx.row(0).copyTo(sMatFx.row(i));
	}
	transpose(sMatFx, sMatFy);

	Mat sMatGaussfft = sMatFx.mul(sMatFx) + sMatFy.mul(sMatFy);
	sMatFx.release();
	sMatFy.release();
	sMatGaussfft.convertTo(sMatGaussfft, CV_32FC1);
	sMatGaussfft     = (-1) * sMatGaussfft / (fS1*fS1);
	exp(sMatGaussfft, sMatGaussfft);
	Mat sMatGaussfftshift = Mat(Size(sMatGaussfft.cols, sMatGaussfft.rows), CV_32FC1);
	fftshift(sMatGaussfft, sMatGaussfftshift, floor(sMatGaussfft.cols/2.0), floor(sMatGaussfft.rows/2.0));
	sMatGaussfft.release();

	Mat sMatOutputTempfft;
	Mat sMatOutputTempifft;
	dft(sMatOutputTemp, sMatOutputTempfft, DFT_COMPLEX_OUTPUT);
	Mat sMatPlanes[2];
	split(sMatOutputTempfft, sMatPlanes);
	sMatPlanes[0] = sMatPlanes[0].mul(sMatGaussfftshift);
	sMatPlanes[1] = sMatPlanes[1].mul(sMatGaussfftshift);
	merge(sMatPlanes, 2, sMatOutputTempfft);
	idft(sMatOutputTempfft, sMatOutputTempifft, DFT_SCALE);
	split(sMatOutputTempifft, sMatPlanes);
	sMatOutputTempfft.release();
	sMatOutputTempifft.release();

	Mat sMatInputIllum = sMatOutputTemp - sMatPlanes[0];
	Mat sMatInputIllumTemp = sMatInputIllum.mul(sMatInputIllum);

	Mat sMatOutputIllumfft;
	Mat sMatOutputIllumifft;
	dft(sMatInputIllumTemp, sMatOutputIllumfft, DFT_COMPLEX_OUTPUT);
	sMatInputIllumTemp.release();
	split(sMatOutputIllumfft, sMatPlanes);
	sMatPlanes[0] = sMatPlanes[0].mul(sMatGaussfftshift);
	sMatPlanes[1] = sMatPlanes[1].mul(sMatGaussfftshift);
	sMatGaussfftshift.release();
	merge(sMatPlanes, 2, sMatOutputIllumfft);
	idft(sMatOutputIllumfft, sMatOutputIllumifft, DFT_SCALE);
	split(sMatOutputIllumifft, sMatPlanes);
	sMatOutputIllumfft.release();
	sMatOutputIllumifft.release();

	Mat sMatLocalStd = abs(sMatPlanes[0]);
	sqrt(sMatLocalStd, sMatLocalStd);
	sMatLocalStd = (0.2) + sMatLocalStd;
	sMatOutputTemp = sMatInputIllum / sMatLocalStd;

	Rect sRectROI(s32Pad, s32Pad, sMatInput.cols, sMatInput.rows);
	sMatOutputTemp(sRectROI).copyTo(sMatOutput);
#if 0
	delete[] pu8Data;
#endif
}


void Gist::fftshift(Mat & sMatInput, Mat & sMatOutput, int s32DimOne, int s32DimTwo)
{
#if DEBUG
	Mat sMatIn = Mat::zeros(Size(5, 5), CV_32FC1);
	randu(sMatIn, Scalar::all(0), Scalar::all(255));
	Mat sMatInput = sMatIn;
	Mat sMatOutput = Mat::zeros(Size(5, 5), CV_32FC1);
	s32DimOne = 2;
	s32DimTwo = 3;
#endif
	if (s32DimOne < 0 || s32DimTwo < 0) {
		cout << "Invalid parameters for fftshift(), kindly check image width and height" << endl;
		return;
	}

	if (s32DimOne == 0 && s32DimTwo == 0) {
		sMatInput.copyTo(sMatOutput);
		return;
	}
	Mat sMatOutputTemp = Mat::zeros(sMatInput.size(), CV_32FC1);
	if (s32DimOne != 0) {
		int s32StartIndex = sMatInput.rows - s32DimOne;
		for (int i = s32StartIndex; i < sMatInput.rows; i++) {
			sMatInput.row(i).copyTo(sMatOutput.row(i - s32StartIndex));
		}
		for (int i = 0; i < s32StartIndex; i++) {
			sMatInput.row(i).copyTo(sMatOutput.row(i + s32DimOne));
		}
	}

	if (s32DimTwo != 0) {
		transpose(sMatOutput, sMatOutputTemp);
		int s32StartIndex = sMatOutputTemp.rows - s32DimTwo;
		for (int i = s32StartIndex; i < sMatOutputTemp.rows; i++) {
			sMatOutputTemp.row(i).copyTo(sMatOutput.row(i - s32StartIndex));
		}
		for (int i = 0; i < s32StartIndex; i++) {
			sMatOutputTemp.row(i).copyTo(sMatOutput.row(i + s32DimTwo));
		}
		transpose(sMatOutput, sMatOutput);
	}
}

Gist::~Gist ( )
{
	delete[] objGistDescriptor.pfGistDescriptor;
	return ;
}		/* -----  end of method Gist::Gist  ----- */
