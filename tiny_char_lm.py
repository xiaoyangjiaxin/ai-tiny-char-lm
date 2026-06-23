import argparse
import json
from pathlib import Path

import numpy as np


TEXT = """
人工智能可以辅助网络工程学习。
子网划分需要理解 IP 地址、CIDR 前缀、子网掩码、网络地址和广播地址。
当我们输入 192.168.1.10/24 时，可以计算出网络地址是 192.168.1.0。
生成式 AI 可以帮助生成代码、解释概念和整理思路。
但是 AI 的输出需要人工验证，不能直接照搬。
学习 LLM 的一个重要思路是预测下一个 token。
在字符级语言模型中，每一个汉字、字母、数字或标点都可以看作一个 token。
模型根据前面的字符，预测下一个字符出现的概率。
训练时会计算 loss，loss 越低，说明模型对训练文本的预测越准确。
本实验使用一个最小字符级语言模型，观察训练前后生成文本和 loss 变化。
"""


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def build_dataset(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = np.array([stoi[ch] for ch in text], dtype=np.int64)
    return data, stoi, itos


def sample_batch(data, batch_size, rng):
    starts = rng.integers(0, len(data) - 1, size=batch_size)
    x = data[starts]
    y = data[starts + 1]
    return x, y


def loss_and_grad(weight, x, y):
    logits = weight[x]
    probs = softmax(logits)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()

    grad_logits = probs
    grad_logits[np.arange(len(y)), y] -= 1
    grad_logits /= len(y)

    grad_weight = np.zeros_like(weight)
    np.add.at(grad_weight, x, grad_logits)
    return loss, grad_weight


def generate(weight, start_id, itos, length, rng):
    current = start_id
    output = [itos[current]]
    for _ in range(length - 1):
        probs = softmax(weight[current][None, :])[0]
        current = int(rng.choice(len(probs), p=probs))
        output.append(itos[current])
    return "".join(output)


def train(args):
    rng = np.random.default_rng(args.seed)
    data, stoi, itos = build_dataset(TEXT)
    vocab_size = len(stoi)
    weight = rng.normal(0, 0.01, size=(vocab_size, vocab_size))

    start_id = data[0]
    before_text = generate(weight, start_id, itos, args.generate_length, rng)
    initial_loss, _ = loss_and_grad(weight, data[:-1], data[1:])

    history = []
    for step in range(1, args.max_iters + 1):
        x, y = sample_batch(data, args.batch_size, rng)
        loss, grad = loss_and_grad(weight, x, y)
        weight -= args.learning_rate * grad
        if step == 1 or step % args.log_interval == 0 or step == args.max_iters:
            full_loss, _ = loss_and_grad(weight, data[:-1], data[1:])
            history.append({"step": step, "train_loss": round(float(full_loss), 4)})

    final_loss, _ = loss_and_grad(weight, data[:-1], data[1:])
    after_text = generate(weight, start_id, itos, args.generate_length, rng)

    result = {
        "run_name": args.run_name,
        "vocab_size": vocab_size,
        "text_length": int(len(data)),
        "batch_size": args.batch_size,
        "max_iters": args.max_iters,
        "learning_rate": args.learning_rate,
        "generate_length": args.generate_length,
        "initial_loss": round(float(initial_loss), 4),
        "final_loss": round(float(final_loss), 4),
        "before_training_text": before_text,
        "after_training_text": after_text,
        "history": history,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{args.run_name}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"运行名称: {args.run_name}")
    print(f"字符表大小 vocab_size: {vocab_size}")
    print(f"训练文本长度 text_length: {len(data)}")
    print(f"参数: batch_size={args.batch_size}, max_iters={args.max_iters}, learning_rate={args.learning_rate}")
    print(f"训练前 loss: {result['initial_loss']}")
    print(f"训练后 loss: {result['final_loss']}")
    print("-" * 60)
    print("训练前生成文本:")
    print(before_text)
    print("-" * 60)
    print("训练后生成文本:")
    print(after_text)
    print("-" * 60)
    print("loss 记录:")
    for item in history:
        print(f"step {item['step']:>4}: loss={item['train_loss']}")
    print(f"结果已保存: {output_path}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="最小字符级语言模型")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--generate-length", type=int, default=120)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

