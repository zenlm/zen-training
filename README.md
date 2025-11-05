---
title: Zen Training
emoji: 🧘
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
license: apache-2.0
hardware: a10g-large
---

# 🧘 Zen Training Space

**Unified Training Platform for All Zen Models**

Train any Zen model with any dataset combination from HuggingFace. Everything runs directly from HF datasets - no local storage needed!

## 🎯 Features

### Supported Models

**Language Models:**
- `zen-nano` (0.6B) - Edge deployment
- `zen-eco` (4B) - Balanced performance
- `zen-omni` (7B) - Multi-task
- `zen-coder` (14B) - Code generation
- `zen-next` (32B) - Frontier performance

**Vision-Language Models:**
- `zen-vl-4b` - Efficient VL with function calling
- `zen-vl-8b` - Enhanced VL capabilities
- `zen-vl-30b` - Maximum VL performance

### Supported Datasets

**Agent Training (ADP):**
- AgentTuning OS/KG/DB (~15k samples)
- Synatra (99k agent trajectories)
- Code Feedback (66k samples)
- Go Browse (27k web interactions)

**Function Calling:**
- xLAM 60k (Salesforce high-quality function calling)

**Instruction Tuning:**
- Alpaca (52k instruction samples)

## 🚀 How to Use

1. **Select Model**: Choose from language or vision-language models
2. **Select Datasets**: Check multiple datasets to combine them
3. **Configure Training**: Set epochs, batch size, learning rate, max samples
4. **Set Output Repo**: Specify HuggingFace repo for trained model
5. **Start Training**: Click the button and monitor logs

## ⚙️ Training Configuration

### Recommended Settings

**4B Models (A10G - 24GB):**
- Batch Size: 1-2
- Max Samples: 10,000-30,000
- Time: 4-8 hours
- Cost: ~$3-5

**8B Models (A100 - 40GB):**
- Batch Size: 2-4
- Max Samples: 30,000-50,000
- Time: 8-12 hours
- Cost: ~$15-20

**32B Models (A100 - 80GB):**
- Batch Size: 1-2
- Max Samples: 50,000-100,000
- Time: 20-30 hours
- Cost: ~$50-80

## 📊 Dataset Combinations

### For Agent Training:
```
ADP Synatra (80%) + xLAM (20%)
= Strong agent + quality function calling
```

### For Code Models:
```
Code Feedback (70%) + Alpaca (30%)
= Code expertise + general instruction following
```

### For VL Models:
```
ADP (all configs) + xLAM
= Complete vision-language agent training
```

## 🔒 Requirements

- HuggingFace Pro account (for GPU access)
- Write access to output repository
- HF_TOKEN secret set in Space settings

## 💡 Tips

1. **Start Small**: Test with 1,000 samples first
2. **Mix Datasets**: Combine complementary datasets for best results
3. **Monitor Logs**: Watch for OOM errors and adjust batch size
4. **Save Often**: Lower save_steps for longer training runs

## 📚 Resources

- **Website**: https://zenlm.org
- **GitHub**: https://github.com/zenlm
- **Models**: https://huggingface.co/zenlm
- **Datasets**:
  - [ADP](https://huggingface.co/datasets/neulab/agent-data-collection)
  - [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)

## 📄 License

Apache 2.0

## 🙏 Citations

```bibtex
@software{zen-training-2025,
  title={Zen Training: Unified Training Platform for Zen Models},
  author={Zen AI Team},
  year={2025},
  url={https://huggingface.co/spaces/zenlm/zen-training}
}

@article{adp2024,
  title={Agent Data Protocol},
  author={NeuLab},
  journal={arXiv preprint arXiv:2510.24702},
  year={2024}
}

@dataset{xlam2024,
  title={xLAM Function Calling Dataset},
  author={Salesforce Research},
  year={2024}
}
```
