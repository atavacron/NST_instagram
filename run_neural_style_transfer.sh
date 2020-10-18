#!/bin/sh

cd ~/Documents/projects/DNN/

source deepenv/bin/activate

#while after the main in neural_style_transfer.py
#making sure this works on dockerfile
############################################# 
#upload=true;
#while $upload;
#do
#if python neural_style_transfer.py; then
#    upload=false
#fi
#done
#############################################

python neural_style_transfer.py