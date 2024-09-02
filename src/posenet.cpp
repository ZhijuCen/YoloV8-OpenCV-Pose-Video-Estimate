
#include "posenet.hpp"

PoseNet::PoseNet(const std::string &onnxModelPath, const cv::Size2f &modelInputShape, const bool useCuda)
{
    modelPath = onnxModelPath;
    inputShape = modelInputShape;
    runWithCuda = useCuda;

    loadOnnxNet();
}

void PoseNet::loadOnnxNet()
{
    net = cv::dnn::readNetFromONNX(modelPath);
    if (runWithCuda)
    {
        std::cout << "Prefer run on CUDA." << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "Prefer run on CPU." << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

std::vector<Pose> PoseNet::operator()(const cv::Mat &inputImg)
{
    cv::Mat modelInput = inputImg;
    if (inputShape.width == inputShape.height)
        modelInput = letterBox(modelInput);
    
    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, inputShape, cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    int rows = outputs[0].size[2];
    int dims = outputs[0].size[1];
    cv::Mat output = outputs[0];
    output = output.reshape(1, dims);
    cv::transpose(output, output);

    float *data = (float *) output.data;

    float x_scale = modelInput.cols / inputShape.width;
    float y_scale = modelInput.rows / inputShape.height;

    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<std::vector<float>>> keypoints;

    for (int i = 0; i < rows; i++)
    {
        float score = data[4];
        if (score > scoreThres)
        {
            confidences.push_back(score);

            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            int x = int((cx - 0.5 * w) * x_scale);
            int y = int((cy - 0.5 * h) * y_scale);
            int width = int(w * x_scale);
            int height = int(h * y_scale);

            boxes.push_back(cv::Rect(x, y, width, height));

            std::vector<std::vector<float>> kps;
            for (int ki = 5; ki < dims; ki += 3)
            {
                float kx = data[ki] * x_scale;
                float ky = data[ki+1] * y_scale;
                float ks = data[ki+2];
                std::vector<float> kp = {kx, ky, ks};
                kps.push_back(kp);
            }
            keypoints.push_back(kps);
        }
        data += dims;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, scoreThres, nmsThres, nms_result);

    std::vector<Pose> poses{};
    for (unsigned long i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];

        Pose pose;
        pose.score = confidences[idx];
        pose.bbox = boxes[idx];
        pose.keypoints = keypoints[idx];

        poses.push_back(pose);
    }
    return poses;
}

cv::Mat PoseNet::letterBox(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int dmax = MAX(col, row);
    cv::Mat dest = cv::Mat::zeros(dmax, dmax, CV_8UC3);
    source.copyTo(dest(cv::Rect(0, 0, col, row)));
    return dest;
}
