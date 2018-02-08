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

#ifndef  __GIST_h__
#define  __GIST_h__
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define DEBUG (0)
/*
 * =====================================================================================
 *        Class:  Gist
 *  Description:  This class implements methods to extract GIST descriptor for Images
 * =====================================================================================
 */
class Gist {
	public:
		Gist();/* constructor */
		~Gist();/* destructor */
		Gist(String strImgFolderPath, String strFileNamesPath, String strGistFileName);/* constructor */

		class Params {
			friend class Gist;
			public:
				vector<int> vecIntOrientation;
			private:
				int s32NumBlocks;
				int s32ImgWHPad;
				int s32NumPad;
				int s32PreFilt;
		};

		class GistDescriptor {
			public:
				int    s32Length;
				float *pfGistDescriptor;
		};

		GistDescriptor* compute(Params *pObjParams);
		GistDescriptor* compute(Mat sMatInput, Params *pObjParams);

	private:
		String         strImgFolderPath;
		String         strGistFilePath;
		String         strFileNamesPath;
		vector<String> vecStrImgPath;
		GistDescriptor objGistDescriptor;
		Params         objParams;
		int            s32GistMode;
		vector<Mat>    vecGaborFilters;
		const int GIST_WIDTH_HEIGHT = 256;

		void createGabor(int s32ImgWidth, int s32ImgHeight);
		void prefilter(Mat& sMatInput, Mat& sMatOutput);
		void fftshift(Mat& sMatInput, Mat& sMatOutput, int s32DimOne, int s32DimTwo);
}; /* -----  end of class Gist  ----- */
#endif   /* ----- #ifndef __gist_h__  ----- */
