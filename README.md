# 🧠 My Generative AI Model (Hugging Face + LoRA)

This project fine-tunes a small language model (GPT-2) using
[Hugging Face Transformers](https://huggingface.co/docs/transformers)
and [PEFT (LoRA)](https://huggingface.co/docs/peft) for efficient training.

## 🔧 Features
- Uses **LoRA adapters** (Parameter Efficient Fine-Tuning)
- Works on **free Colab / local GPU**
- Pushes model to **Hugging Face Hub**

---

## 🧰 Setup

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Format

Place your dataset inside `data/train.jsonl` like:

```json
{"prompt": "Write a haiku about AI:\n", "completion": "Thinking machine hums,\nPatterns bloom in silent code,\nDreams of logic grow.\n"}
{"prompt": "Explain recursion simply:\n", "completion": "Recursion is when a function calls itself until it finishes a task.\n"}
```

---

## 🚀 Train

```bash
python train_lora.py
```

After training, your adapter is saved in `./gpt2-lora/`.

To push it to Hugging Face:

```bash
huggingface-cli login
python train_lora.py --push_to_hub True --hub_model_id YOUR_USERNAME/my-generative-ai
```

---

## 💬 Inference

```bash
python inference.py
```

---

## 📚 Resources
- [Transformers Docs](https://huggingface.co/docs/transformers)
- [PEFT / LoRA Docs](https://huggingface.co/docs/peft)
