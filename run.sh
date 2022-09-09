#!/bin/bash

if [[ $OSTYPE == 'darwin'* ]]; then
    source tool/osx.sh
fi

mkdir build 
cd build 
cmake ..
cmake --build .
cd ..
./build/ClothSimulation
python3.9 tool/render.py 
convert -delay 10 -loop 0 -limit memory 2GiB tool/pngs/*.png result/animation.gif
rm -r tool/pngs