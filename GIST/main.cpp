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
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include "src/gist.cpp"
//#include <vld.h>

//OpenCV libraries that might be used.
#if 0
#pragma comment(lib, "opencv_core2413d.lib") // core functionalities
#pragma comment(lib, "opencv_highgui2413d.lib") //GUI
#pragma comment(lib, "opencv_imgproc2413d.lib") // Histograms, Edge detection
#else
#pragma comment(lib, "opencv_core2413.lib") // core functionalities
#pragma comment(lib, "opencv_highgui2413.lib") //GUI
#pragma comment(lib, "opencv_imgproc2413.lib") // Histograms, Edge detection
#endif

#define GIST_VERSION "Gist 1.0 Beta"
using namespace std;
using namespace cv;

String pathAppend(const String& strPathOne, const String& strPathTwo) {

	char cSep = '/';
	String strTmpPath = strPathOne;

#ifdef _WIN32
	cSep = '\\';
#endif

	if (strPathOne[strPathOne.length()] != cSep) { // Need to add a
		strTmpPath += cSep;                // path separator
		return(strTmpPath + strPathTwo);
	} else {
		return(strPathOne + strPathTwo);
	}
}

void help()
{
	cout << endl
		<< GIST_VERSION
		<< endl
		<< "------------"
		<< endl
		<< "Usage:"
		<< endl << endl
		<< ">>gist.exe -i [input directory] -o [output directory]"
		<< endl << endl
		<< "[  input directory ]  - input directory where images are present"
		<< endl
		<< "[ output directory ]  - output directory where file names and respective gist vector are dumped"
		<< endl
		<< "                      into gist.txt and filenames.txt"
		<< endl
		<< "gist.txt              - each line contain gist vector for each file in filenames.txt"
		<< endl << endl;

}
/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  main
 *  Description:  
 * =====================================================================================
 */
int main ( int argc, char *argv[] )
{
	vector<String> cmdArgs;
	if (argc == 5) {
		cmdArgs.assign(argv + 1, argv + argc);
	} else {
		help();
		return EXIT_FAILURE;
	}

	String strInFolderPath; String strOutFolderPath;
	if (!cmdArgs.at(0).compare("-i")) {
		strInFolderPath = cmdArgs.at(1);
	}

	if (!cmdArgs.at(2).compare("-o")) {
		strOutFolderPath = cmdArgs.at(3);
	}

	String strGistFilePath  = pathAppend(strOutFolderPath, "gist.txt");
	String strFileNamesPath = pathAppend(strOutFolderPath, "filenames.txt");

	vector<String> vecStrImgPath;
	glob(strInFolderPath, vecStrImgPath);

	ofstream ofstreamFileNames;
	ofstream ofstreamGist;
	ofstreamFileNames.open(strFileNamesPath);
	ofstreamGist.open(strGistFilePath);
	float fTotalTime = 0;
	for (int i = 0; i < vecStrImgPath.size(); i++) {
		int percent = (int)((i+1)*100.0/vecStrImgPath.size());
		Mat sMatInput = imread(vecStrImgPath.at(i), IMREAD_COLOR);
		if (sMatInput.empty()) {
			cout << "\r" << percent << "% completed:: Total time passed in seconds: " << fTotalTime;
			continue;
		}
		ofstreamFileNames << vecStrImgPath.at(i) << endl;

		Gist *pObjGist = new Gist();
		//Set the parameters for computing gist
		Gist::Params *pObjParams = new Gist::Params();
		vector<int> vecS32Or(4, 8);// 4 scales, 8 orientations
		pObjParams->vecIntOrientation = vecS32Or;
		double t = (double)getTickCount();
		Gist::GistDescriptor *pObjGistDesc = pObjGist->compute(sMatInput, pObjParams);
		vector<float> vecGistDesc(pObjGistDesc->pfGistDescriptor, pObjGistDesc->pfGistDescriptor + pObjGistDesc->s32Length);
		copy(vecGistDesc.begin(), vecGistDesc.end(), ostream_iterator<float>(ofstreamGist, " "));
		ofstreamGist << endl;

		t = ((double)getTickCount() - t) / getTickFrequency();
		fTotalTime = fTotalTime + t;
		cout << "\r" << percent << "% completed:: Total time passed in seconds: " << fTotalTime;
		delete pObjParams;
		delete pObjGist;
	}
	ofstreamGist.close();
	ofstreamFileNames.close();
	cout << endl;


	return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
