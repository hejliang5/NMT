# Chinese-English-Machine-Translation (RNN-vs-Transformer)
This is a project for NLP &amp;&amp; LLM course.

---
## Preparation

Pretrained Language Model Weight: https://huggingface.co/google-t5/t5-base/tree/main
Download and place in the main directory, like /t5-base,
Download checkpoints and place in the main directory, like /checkpoints/transformer_ablation/base/best_model.pt

## One-Click Inference

Translate one sentence (default uses Transformer base checkpoint):

```bash
cd /your path to the main directory
python inference.py --text "我喜欢自然语言处理"
```

Translate a file (one sentence per line):

```bash
python inference.py --input input_zh.txt --output pred_en.txt
```

Switch to an RNN checkpoint:

```bash
python inference.py --arch rnn --checkpoint checkpoints/rnn_ablation/align_dot/best_model.pt --text "我和你"
```

---
