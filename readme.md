# Chinese–English Machine Translation (RNN vs Transformer)
 
 This is project of NLP && LLM course.
---

## One-Click Inference

Translate one sentence (default uses Transformer base checkpoint):

```bash
cd /your/path/to/main/directory
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
