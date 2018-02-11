cd GIST

make clean
make IMAGEPATH=/data/amrits/ugp/images/

cd ../semantic_features

python test.py --test_img_file /data/amrits/ugp/image_files.txt --test_img_directory /data/amrits/ugp/images/ --weights_encoder /data/amrits/ugp/weights/encoder_best.pth --weights_decoder /data/amrits/ugp/weights/decoder_best.pth 

cd ..

python train.py /data/amrits/ugp/image_files.txt /data/amrits/ugp/images/

