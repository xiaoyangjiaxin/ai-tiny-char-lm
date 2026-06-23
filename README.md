# 作业一：LLM 最小代码实践

本文件夹用于完成“运行最小 LLM / 字符级语言模型，理解训练闭环”的作业。

## 文件说明

- `tiny_char_lm.py`：最小字符级语言模型代码。
- `results/`：训练结果 JSON 和终端记录。
- `作业一实践报告.md`：作业报告。
- `运行截图.png`：运行结果截图。

## 运行方式

基线运行：

```bash
python tiny_char_lm.py --run-name baseline --batch-size 32 --max-iters 500 --learning-rate 1.0
```

参数修改运行：

```bash
python tiny_char_lm.py --run-name modified --batch-size 64 --max-iters 900 --learning-rate 0.8
```

## 实验目标

观察训练前后生成文本和 loss 的变化，并解释 token、loss、next-token prediction。

