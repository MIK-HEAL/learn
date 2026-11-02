明白了，你是想找“从窗口创建开始、逐步走到类似 backflip_AI”的现成教程，而不是让我直接改代码。我会按学习路线筛选教程，并区分窗口/OpenGL、物理模拟、强化学习和 DeepMimic 四部分。

Searched the web for `推荐从 GLFW GLEW OpenGL 创建窗口开始，到 3D 物理模拟、DeepMimic humanoid PPO 后空翻的详细教程和课程，优先官方...`
Completed with input: {
  "query": "推荐从 GLFW GLEW OpenGL 创建窗口开始，到 3D 物理模拟、DeepMimic humanoid PPO 后空翻的详细教程和课程，优先官方文档、LearnOpenGL、Bullet、DeepMimic、PPO 教程"
}

抱歉，刚才理解偏了。我不会把“找教程”继续做成代码任务；另外我会撤销刚才对 `window.cpp` 的修改，避免留下你没要求的改动。

Edit 

明白了，刚才是我理解错了，已经撤销对 `window.cpp` 的修改。

你要找的不是单个“后空翻教程”，而是类似下面这种完整学习路线：

1. **窗口和 OpenGL 基础**  
   [LearnOpenGL：Getting Started](https://learnopengl.com/Getting-started/OpenGL)  
   从 GLFW 创建窗口、GLEW、输入、Shader、三角形开始，最符合你的需求。

2. **OpenGL 官方窗口文档**  
   [GLFW Documentation](https://www.glfw.org/docs/latest/)  
   适合查窗口、键盘、鼠标、OpenGL Context。

3. **3D 物理模拟**  
   [Bullet Physics](https://github.com/bulletphysics/bullet3)  
   学刚体、碰撞、关节、重力和角色物理。

4. **人形动作模仿**  
   [DeepMimic](https://github.com/xbpeng/DeepMimic)  
   这是和 `backflip_AI` 最接近的项目，包含物理人形、动作模仿和强化学习。

5. **PPO 强化学习**  
   [OpenAI Spinning Up：PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

推荐顺序：

```text
LearnOpenGL
→ GLFW / GLEW
→ Shader 和 3D 摄像机
→ Bullet 物理
→ 人形关节和动作
→ PPO
→ DeepMimic / backflip_AI
```

目前没有一套教程能完整覆盖“窗口创建 → OpenGL → 物理人形 → PPO 后空翻”；通常需要把这些教程串起来学习。最适合作为起点的是 **LearnOpenGL**。







有，但“从窗口一路喂到后空翻 AI”的单一教程很少。我去找更接近“按步骤运行、改参数、逐步复现”的项目，而不是只列理论文档。

Searched the web for `beginner step by step tutorial humanoid physics reinforcement learning backfl...`
Completed with input: {
  "query": "beginner step by step tutorial humanoid physics reinforcement learning backflip DeepMimic PyBullet repository walkthrough"
}

“Get file or directory contents”失败
Completed with input: {
  "owner": "digital-humans-project",
  "repo": "deep-mimic-pybullet",
  "path": "README.md",
  "ref": "refs/heads/master"
}
Tool execution failed: MCP server 'github-mcp-server': failed to resolve git reference: failed to get final reference for "refs/heads/master": GET https://api.github.com/repos/digital-humans-project/deep-mimic-pybullet/git/ref/heads/master: 404 Not Found []

Get file or directory contents
Completed with input: {
  "owner": "xbpeng",
  "repo": "MimicKit",
  "path": "README.md",
  "ref": "refs/heads/main"
}

Get file or directory contents
Completed with input: {
  "owner": "xbpeng",
  "repo": "DeepMimic",
  "path": "README.md",
  "ref": "refs/heads/master"
}

Get file or directory contents
Completed with input: {
  "owner": "digital-humans-project",
  "repo": "deep-mimic-pybullet",
  "path": "/",
  "fields": [
    "name",
    "path",
    "type",
    "size"
  ]
}

Read , lines 1 to 120
Completed with input: {
  "path": "C:\\Users\\MIKHEAL\\AppData\\Local\\Temp\\1793540959132-copilot-tool-output-5ff09d686e1f4331bed88a12ee239ee4.txt",
  "view_range": [
    1,
    120
  ]
}

Read , lines 1 to 120
Completed with input: {
  "path": "C:\\Users\\MIKHEAL\\AppData\\Local\\Temp\\1793540959257-copilot-tool-output-256395fb80e544e695a7b82229aaa0bf.txt",
  "view_range": [
    1,
    120
  ]
}

有，最接近“喂饭式”的是下面这个：

## 首选：Python 版 DeepMimic

[deep-mimic-pybullet](https://github.com/digital-humans-project/deep-mimic-pybullet)

它比 `backflip_AI` 更适合入门：

- Python 为主，不需要先读大量 C++
- 使用 PyBullet，不需要自己写 OpenGL 窗口
- 有 `conf/`、`envs/`、`model/`、`scripts/` 等清晰目录
- 可以先运行已有模型，再看训练
- 结构仍然包含：人形、动作数据、物理环境、PPO 训练

建议学习顺序：

```text
1. 运行预训练模型
2. 看 envs/：机器人和物理环境
3. 看 conf/：训练参数
4. 看 model/：神经网络和 PPO
5. 修改动作或奖励
6. 最后再训练后空翻
```

## 第二选择：MimicKit

[MimicKit](https://github.com/xbpeng/MimicKit)

这是 DeepMimic 作者的新项目，代码比原版更现代，但依赖比较重，需要 Isaac Gym、Isaac Lab 或 Newton，不太适合作为第一步。

它的文档相对完整：

- [DeepMimic 使用说明](https://github.com/xbpeng/MimicKit/blob/main/docs/README_DeepMimic.md)
- [Starter Guide](https://arxiv.org/abs/2510.13794)

## 不建议一开始直接学

[原版 DeepMimic](https://github.com/xbpeng/DeepMimic)

它已经被作者标记为 deprecated，而且依赖：

- C++
- Bullet
- Eigen
- OpenGL
- SWIG
- MPI
- TensorFlow 1.x

Windows 配置会比较痛苦，适合后面用来理解原理，不适合零基础入门。

## 最适合你的实际路线

```text
Python 基础
↓
PyBullet 基础：小球、方块、关节
↓
deep-mimic-pybullet：运行人形动作
↓
理解 PPO 和奖励函数
↓
训练简单动作：站立、走路
↓
训练后空翻
↓
再阅读 backflip_AI 的 C++ 源码
```

你的目标项目 [backflip_AI](https://github.com/kavan010/backflip_AI) 本身更像“作者已经把教程跳过后的最终工程”，不是特别适合从窗口开始照着学。

如果只选一个，我建议：

> **先学 `deep-mimic-pybullet`，不要先碰 `backflip_AI`。**