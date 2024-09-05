
# Yolo OpenCV Pose Video Estimate

## Setup

### Get Weigth file from [YOLOv8 Repo](https://github.com/ultralytics/ultralytics)

```sh
# In python env/virtualenv
python -m pip install ultralytics
yolo export task=pose model=yolov8x-pose.pt format=onnx
```

### Configure and Build

```sh
# Configuration and install packages in 3h
cmake --preset=default-cuda
# Build
cmake --build ./build --parallel 14
```

Or, if prefer run on CPU

```sh
rm vcpkg.json
mv vcpkg-cpu.json vcpkg.json
cmake --preset=default
cmake --build ./build --parallel 14
```

## Usage

```sh
./build/main -m yolov8x-pose.onnx -i input_video.mp4 -o output_video.mp4 -s 0.25 -n 0.4
```
