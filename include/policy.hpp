#pragma once

#include <string>
#include <vector>

/// 由 py/train.py 导出的多层感知机策略。
///
/// 网络结构固定为：若干 (Linear -> tanh)，最后一层同样带 tanh，
/// 因此输出天然落在 [-1, 1]，可以直接作为 Ragdoll 的动作。
class Policy {
public:
    /// 从 train.py 导出的二进制文件加载，失败返回 false
    bool load(const std::string& path);

    bool valid() const { return !layers_.empty(); }
    int actionCount() const;
    int observationCount() const;

    /// observation 需要 observationCount() 个元素，actions 需要 actionCount() 个
    void forward(const float* observation, float* actions) const;

private:
    struct Layer {
        int in{};
        int out{};
        std::vector<float> weights;
        std::vector<float> bias;
    };

    std::vector<Layer> layers_;
    std::vector<float> mean_;
    std::vector<float> inverseStd_;
    mutable std::vector<float> bufferA_;
    mutable std::vector<float> bufferB_;
};
