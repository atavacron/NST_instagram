#!/bin/sh

cd ~/Documents/projects/DNN/

source deepenv/bin/activate

upload=true;

while $upload;
do
if python neural_style_transfer.py; then
    upload=false
fi
done
