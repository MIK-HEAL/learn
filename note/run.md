# 只看水平，不训练（确定性策略）
python py/train.py --device cuda --eval-only py/policy.bin --eval-episodes 10

# 继续训（自动继承 actor + 归一化 + 噪声水平，并保护原文件）
python py/train.py --device cuda --workers 4 --envs 4 --resume py/policy.bin --steps 2000000



# 看效果
build\msvc\bin\Release\window.exe --watch py\policy.bin