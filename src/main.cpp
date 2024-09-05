
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "posenet.hpp"
#include "pose.hpp"
#include "imgprocess.hpp"

namespace po = boost::program_options;

bool create_options(int argc, char** argv, po::variables_map &vm);

int main(int argc, char** argv)
{
    po::variables_map vm;
    bool options_created = create_options(argc, argv, vm);
    if (!options_created)
        return 1;
    PoseNet net(vm["model-path"].as<std::string>());
    net.setScoreThres(vm["score-thres"].as<float>())
        .setNmsThres(vm["nms-thres"].as<float>());
    cv::VideoCapture cap(vm["input-video"].as<std::string>());
    if (!cap.isOpened())
    {
        std::cerr << "Cannot Open Video File." << std::endl;
        return 1;
    }
    int vid_width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int vid_height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double vid_fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter vid_writer(vm["output-video"].as<std::string>(),
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
        vid_fps, cv::Size(vid_width, vid_height));
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        std::vector<Pose> poses = net(frame);
        for (auto pose : poses)
        {
            drawBBox(frame, pose.bbox);
            drawKeypoints(frame, pose);
        }
        vid_writer << frame;
    }
    vid_writer.release();
    return 0;
}

bool create_options(int argc, char** argv, po::variables_map &vm)
{
    using namespace std;
    po::options_description desc("Options");
    desc.add_options()
        ("help", "produce help message.")
        ("model-path,m", po::value<string>()->default_value("yolov8n-pose.onnx"), "Path to onnx model.")
        ("input-video,i", po::value<string>()->default_value("input_video.mp4"), "Path to input video.")
        ("output-video,o", po::value<string>()->default_value("output_video.mp4"), "Path to output video.")
        ("score-thres,s", po::value<float>()->default_value(0.3), "Score threshold to show pose.")
        ("nms-thres,n", po::value<float>()->default_value(0.8), "IOU threshold to execute NMS.");
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help"))
    {
        cout << desc << endl;
        return false;
    }
    return true;
}