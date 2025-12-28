# Chinese-English-Machine-Translation (RNN-vs-Transformer)
This is a project for NLP &amp;&amp; LLM course.

---
## Preparation

Pretrained Language Model Weight: https://huggingface.co/google-t5/t5-base/tree/main

Download and place in the main directory, like /t5-base

Download checkpoints and place in the main directory, like /checkpoints/best_model.pt

Checkpoints file is placed in college's public server.

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
---
