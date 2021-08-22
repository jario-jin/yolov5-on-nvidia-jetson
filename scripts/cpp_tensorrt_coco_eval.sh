cd build
./yolov5 -d yolov5s.engine ../val2017
cd ..
python3 scripts/convert_id2str.py
