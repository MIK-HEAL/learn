#pragma once

#include <box2d/box2d.h>
#include <glm/vec2.hpp>

#include <array>
#include <random>
#include <vector>

/// 布娃娃（ragdoll）物理模型：只处理物理，不依赖 OpenGL。
///
/// 它同时被两个目标使用：
///   * rl_env 动态库（无渲染，供 Python 强化学习训练）
///   * window 可执行程序（OpenGL 绘制与策略回放）
/// 这样训练与回放的运动学完全一致。
class Ragdoll {
public:
    struct Part {
        b2BodyId body{};
        glm::vec2 halfSize{};
    };

    /// 躯干 + 头 + 左右大臂/小臂 + 左右大腿/小腿
    static constexpr int kPartCount = 10;
    /// 颈部 1 + 肩 2 + 肘 2 + 髋 2 + 膝 2
    static constexpr int kJointCount = 9;
    /// 除颈部以外的关节都可以被电机驱动
    static constexpr int kActuatorCount = 8;

    /// 观测维度 = 躯干 7 + 每部件相对位姿 4*10 + 关节角 9 + 部件角速度 10
    static constexpr int kObsSize = 7 + kPartCount * 4 + kJointCount + kPartCount;

    /// 地面顶面所在的世界坐标 y（与 Ground 的绘制保持一致）
    static constexpr float kGroundTopY = -4.0f;
    /// 默认生成时的竖直偏移：让脚底刚好踩在地面上
    static constexpr float kSpawnOffset = -2.7f;

    /// 动作 [-1, 1] 映射到的最大目标角速度（rad/s）
    static constexpr float kMotorSpeedScale = 8.0f;
    /// 关节电机最大力矩（N*m）
    static constexpr float kMotorTorque = 160.0f;

    /// 站立时躯干中心的世界坐标高度（= 2.6 + kSpawnOffset），用于奖励塑形
    static constexpr float kStandHeight = -0.1f;

    /// 摔倒判定阈值：躯干比站姿低 1.2 米就算失败（跪姿约低 1.5 米以上）
    static constexpr float kFallHeight = kStandHeight - 1.2f;
    static constexpr float kFallUpY = 0.35f;

    explicit Ragdoll(b2WorldId world, float verticalOffset = kSpawnOffset);
    ~Ragdoll();

    Ragdoll(const Ragdoll&) = delete;
    Ragdoll& operator=(const Ragdoll&) = delete;

    /// 设置各驱动关节的目标角速度，actions 取值范围 [-1, 1]
    void setActions(const float* actions);

    /// 给所有部件叠加随机初始速度，用于训练时的状态随机化
    void randomize(std::mt19937& rng, float linearScale, float angularScale);

    /// 写出 kObsSize 个 float 的观测
    void getObservation(float* observation) const;

    /// 根据动作与当前状态计算即时奖励
    float reward(const float* actions) const;

    /// 是否已经摔倒
    bool fallen() const;

    glm::vec2 basePosition() const;
    /// 躯干"向上"方向在世界坐标 y 上的分量，1 表示完全直立
    float baseUprightness() const;

    const std::vector<Part>& parts() const { return parts_; }
    const std::vector<b2JointId>& joints() const { return joints_; }

private:
    Part createPart(glm::vec2 position, glm::vec2 halfSize);
    b2JointId connect(b2BodyId first, b2BodyId second, glm::vec2 anchor);

    b2WorldId world_{};
    std::vector<Part> parts_;
    std::vector<b2JointId> joints_;
    std::array<b2JointId, kActuatorCount> actuatorJoints_{};
    /// 当前 / 上一步动作，用于惩罚动作突变（抖动）
    std::array<float, kActuatorCount> actions_{};
    std::array<float, kActuatorCount> previousActions_{};
};
