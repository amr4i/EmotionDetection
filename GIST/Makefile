all: 
	g++ -std=c++11 -o gist main.cpp -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_hdf -lopencv_rgbd -lopencv_face -lopencv_photo -lopencv_optflow -lopencv_freetype -lopencv_reg -lopencv_ccalib -lopencv_bioinspired -lopencv_dpm -lopencv_saliency -lopencv_stereo -lopencv_xobjdetect -lopencv_objdetect -lopencv_surface_matching -lopencv_fuzzy -lopencv_ximgproc -lopencv_img_hash -lopencv_xfeatures2d -lopencv_shape -lopencv_line_descriptor -lopencv_dnn_objdetect -lopencv_aruco -lopencv_bgsegm -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_video -lopencv_ml -lopencv_structured_light -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_phase_unwrapping -lopencv_flann -lopencv_hfs -lopencv_plot -lopencv_xphoto -lopencv_imgproc -lopencv_core
	mkdir ../tmp
	./gist -i ${IMAGEPATH} -o ../tmp/
	python ../preprocess_gist_files.py 
	rm -rf ../tmp

clean: 
	rm gist
	rm -rf ../tmp