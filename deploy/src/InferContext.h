#pragma once
#include <torch/script.h> // One-stop header.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Timer
{
public:
    Timer()
        : begTime(cv::getTickCount()), endTime(cv::getTickCount()),
          freq(cv::getTickFrequency()), elapsedTime(0)
    {};

    void begin()
    {
        begTime = cv::getTickCount();
    };

    void end()
    {
        endTime = cv::getTickCount();
        elapsedTime = double(endTime - begTime) / freq;
    };

    double elapsed() const
    {
        return elapsedTime;
    }

private:
    long long int begTime, endTime;
    double freq, elapsedTime;
};

// https://github.com/pytorch/pytorch/issues/14219

inline cv::Mat prepareImageBatch(const std::vector<cv::Mat>& images) 
{
    int num = (int)images.size();
    if (num == 0)
        return cv::Mat();

    int rows = images[0].rows;
    int cols = images[0].cols;
    for (int i = 1; i < num; i++) 
    {
        if (rows != images[i].rows ||
            cols != images[i].cols)
            return cv::Mat();
    }
    // std::cout << rows << ", " << cols << "\n";

    int dims[4] = {0};
    dims[0] = num, dims[1] = 3, dims[2] = rows, dims[3] = cols;
    cv::Mat ret(4, dims, CV_32FC1);

    cv::Mat imageChannels[3];
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float stdDev[3] = {0.229f, 0.224f, 0.225f};
    cv::Range ranges[4];
    int newDims[2];

    for (int k = 0; k < num; k++) 
    {
        cv::split(images[k], imageChannels);
        for (int i = 0; i < 3; i++) 
        {
            ranges[0] = cv::Range(k, k + 1);
            ranges[1] = cv::Range(i, i + 1);
            ranges[2] = cv::Range::all();
            ranges[3] = cv::Range::all();
            cv::Mat dst = ret(ranges);
            newDims[0] = rows;
            newDims[1] = cols;
            cv::Mat dstReshape = dst.reshape(1, 2, newDims);
            dstReshape = imageChannels[i] - mean[i];
            dstReshape *= (1.0f / stdDev[i]);
        }
    }

    return ret;
}

inline at::Tensor prepareImageTensor(const std::vector<cv::Mat>& images, 
    int width, int height)
{
    std::vector<cv::Mat> imagesReady;
    for (const cv::Mat& image : images)
    {
        cv::Mat cvtImage, resizedImage, image32f;
        cv::cvtColor(image, cvtImage, cv::COLOR_BGR2RGB);
        cv::resize(cvtImage, resizedImage, cv::Size(width, height));
        resizedImage.convertTo(image32f, CV_32F, 1.0 / 255);
        imagesReady.push_back(image32f);
    }
    cv::Mat normImage = prepareImageBatch(imagesReady);
    at::Tensor output = torch::from_blob(normImage.ptr<float>(), {(int)images.size(), 3, height, width});
    return output.clone();
}

struct InferContext 
{
    InferContext(const std::string& modelPath, bool useCuda, int height, int width) 
    {
        module_ = torch::jit::load(modelPath);
        useCuda_ = useCuda;
        if (useCuda_)
            module_->to(at::kCUDA);
        height_ = height;
        width_ = width;

        accTotalTime_ = 0;
        accPrepareTime_ = 0;
        accForwardTime_ = 0;
        inferCount_ = 0;
    }

    struct Result 
    {
        Result(int index = 0, float confidence = 0.f) :
            index_(index), confidence_(confidence) {}

        void printLine() const 
        {
            printf("index: %d, confidence: %f\n", index_, confidence_);
        }

        int index_;
        float confidence_;
    };

    void infer(const std::vector<cv::Mat>& images, std::vector<Result>& results) 
    {
        results.clear();
        int numImages = (int)images.size();
        if (numImages == 0)
        {
            printf("Num images is 0, return\n");
            return;
        }

        totalTimer_.begin();

        timer_.begin();
        tensorBuf_ = prepareImageTensor(images, width_, height_);
        timer_.end();
        accPrepareTime_ += timer_.elapsed();
        // printf("%ld, %ld, %ld, %ld\n", tensor_buf_.size(0), tensor_buf_.size(1), 
        //     tensor_buf_.size(2), tensor_buf_.size(3));
        
        inputs_.clear();
        if (useCuda_)
            inputs_.push_back(tensorBuf_.to(at::kCUDA));
        else
            inputs_.push_back(tensorBuf_);
        // std::cout << "data prepare ok\n";

        // Execute the model and turn its output into a tensor.  
        timer_.begin();      
        at::Tensor output = module_->forward(inputs_).toTensor();
        at::Tensor outputSoftmax = useCuda_ ? at::softmax(output, 1).to(at::kCPU) : at::softmax(output, 1);        
        timer_.end();
        accForwardTime_ += timer_.elapsed();

        // std::cout << output.dim() << '\n';
        // std::cout << output.sizes() << '\n';
        // std::cout << output.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';
        // std::cout << output_softmax.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';
        
        int numClasses = (int)outputSoftmax.size(1);
        at::TensorAccessor<float, 2> accessor = outputSoftmax.accessor<float, 2>();
        results.resize(numImages);
        for (int row = 0; row < numImages; row++) 
        {
            float maxVal = 0;
            int maxIndex = 0;
            for (int col = 0; col < numClasses; col++) 
            {
                if (accessor[row][col] > maxVal)
                {
                    maxVal = accessor[row][col];
                    maxIndex = col;
                }
            }
            results[row].index_ = maxIndex;
            results[row].confidence_ = maxVal;
        }

        inferCount_++;
        totalTimer_.end();
        accTotalTime_ += totalTimer_.elapsed();
    }

    int getInferCount() const 
    {
        return inferCount_;
    }

    double getAvgInferTime() const 
    {
        return inferCount_ == 0 ? 0 : accTotalTime_ / inferCount_;
    }

    double getAvgPrepareTime() const 
    {
        return inferCount_ == 0 ? 0 : accPrepareTime_ / inferCount_;
    }

    double getAvgForwardTime() const 
    {
        return inferCount_ == 0 ? 0 : accForwardTime_ / inferCount_;
    }

    std::shared_ptr<torch::jit::script::Module> module_;
    bool useCuda_;
    int height_;
    int width_;
    at::Tensor tensorBuf_;
    std::vector<torch::jit::IValue> inputs_;
    Timer timer_;
    Timer totalTimer_;
    double accTotalTime_;
    double accPrepareTime_;
    double accForwardTime_;
    int inferCount_;
};
