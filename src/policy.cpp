#include "policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
constexpr std::int32_t kMagic = 0x504C4D31; // 'P','L','M','1'

template <typename T>
bool readValue(std::ifstream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(stream);
}

bool readFloats(std::ifstream& stream, std::vector<float>& values, int count) {
    values.resize(static_cast<std::size_t>(count));
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(sizeof(float) * static_cast<std::size_t>(count)));
    return static_cast<bool>(stream);
}
} // namespace

bool Policy::load(const std::string& path) {
    // 事务式加载：只有完整读完才替换内部状态，
    // 这样热重载时遇到「文件正被写入」可以安全地保留旧策略。
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        std::cerr << "无法打开策略文件: " << path << '\n';
        return false;
    }

    std::int32_t magic = 0;
    std::int32_t observationCount = 0;
    std::int32_t actionCountValue = 0;
    std::int32_t layerCount = 0;
    if (!readValue(stream, magic) || !readValue(stream, observationCount) ||
        !readValue(stream, actionCountValue) || !readValue(stream, layerCount)) {
        std::cerr << "策略文件头损坏: " << path << '\n';
        return false;
    }
    if (magic != kMagic || observationCount <= 0 || actionCountValue <= 0 || layerCount <= 0) {
        std::cerr << "策略文件格式不匹配: " << path << '\n';
        return false;
    }

    std::vector<float> mean;
    std::vector<float> stddev;
    if (!readFloats(stream, mean, observationCount) ||
        !readFloats(stream, stddev, observationCount)) {
        std::cerr << "策略文件缺少归一化参数: " << path << '\n';
        return false;
    }

    std::vector<Layer> layers;
    layers.reserve(static_cast<std::size_t>(layerCount));
    for (std::int32_t index = 0; index < layerCount; ++index) {
        Layer layer;
        if (!readValue(stream, layer.in) || !readValue(stream, layer.out)) {
            std::cerr << "策略文件层信息损坏: " << path << '\n';
            return false;
        }
        if (layer.in <= 0 || layer.out <= 0) {
            std::cerr << "策略文件层维度非法: " << path << '\n';
            return false;
        }
        if (!readFloats(stream, layer.weights, layer.in * layer.out) ||
            !readFloats(stream, layer.bias, layer.out)) {
            std::cerr << "策略文件层参数损坏: " << path << '\n';
            return false;
        }
        layers.push_back(std::move(layer));
    }

    if (layers.front().in != observationCount || layers.back().out != actionCountValue) {
        std::cerr << "策略文件维度与网络不一致: " << path << '\n';
        return false;
    }

    mean_ = std::move(mean);
    inverseStd_.resize(stddev.size());
    for (std::size_t index = 0; index < stddev.size(); ++index) {
        inverseStd_[index] = 1.0f / std::max(stddev[index], 1e-6f);
    }
    layers_ = std::move(layers);
    // 两个缓冲区都按「最大层维度」分配：forward 每算完一层就交换两者，
    // 否则首层输出维度大于输入维度时会写出界。
    std::size_t bufferSize = static_cast<std::size_t>(observationCount);
    for (const Layer& layer : layers_) {
        bufferSize = std::max(bufferSize, static_cast<std::size_t>(layer.out));
    }
    bufferA_.assign(bufferSize, 0.0f);
    bufferB_.assign(bufferSize, 0.0f);
    return true;
}

int Policy::actionCount() const {
    return layers_.empty() ? 0 : layers_.back().out;
}

int Policy::observationCount() const {
    return static_cast<int>(mean_.size());
}

void Policy::forward(const float* observation, float* actions) const {
    if (layers_.empty() || observation == nullptr || actions == nullptr) {
        return;
    }

    const std::size_t inputSize = mean_.size();
    for (std::size_t index = 0; index < inputSize; ++index) {
        bufferA_[index] = (observation[index] - mean_[index]) * inverseStd_[index];
    }

    for (const Layer& layer : layers_) {
        for (int outIndex = 0; outIndex < layer.out; ++outIndex) {
            const auto row = static_cast<std::size_t>(outIndex) * static_cast<std::size_t>(layer.in);
            float sum = layer.bias[static_cast<std::size_t>(outIndex)];
            for (int inIndex = 0; inIndex < layer.in; ++inIndex) {
                sum += layer.weights[row + static_cast<std::size_t>(inIndex)] *
                       bufferA_[static_cast<std::size_t>(inIndex)];
            }
            bufferB_[static_cast<std::size_t>(outIndex)] = std::tanh(sum);
        }
        std::swap(bufferA_, bufferB_);
    }

    const std::size_t outputSize = static_cast<std::size_t>(layers_.back().out);
    std::copy_n(bufferA_.begin(), outputSize, actions);
}
