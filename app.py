"""
Zen Training Space - Unified Training for All Zen Models
Train any Zen model with any dataset combination from HuggingFace
"""

import os
import gradio as gr
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets
import json
from typing import List, Dict

# Model configurations
MODELS = {
    "Language Models": {
        "zen-nano-0.6b": {
            "hf_id": "zenlm/zen-nano-0.6b",
            "type": "language",
            "size": "0.6B",
            "context": "32K"
        },
        "zen-eco-4b-instruct": {
            "hf_id": "zenlm/zen-eco-4b-instruct",
            "type": "language",
            "size": "4B",
            "context": "32K"
        },
        "zen-eco-4b-agent": {
            "hf_id": "zenlm/zen-eco-4b-agent",
            "type": "language",
            "size": "4B",
            "context": "32K"
        },
        "zen-omni-7b": {
            "hf_id": "zenlm/zen-omni-7b",
            "type": "language",
            "size": "7B",
            "context": "32K"
        },
        "zen-coder-14b": {
            "hf_id": "zenlm/zen-coder-14b",
            "type": "language",
            "size": "14B",
            "context": "128K"
        },
        "zen-next-32b": {
            "hf_id": "zenlm/zen-next-32b",
            "type": "language",
            "size": "32B",
            "context": "32K"
        },
    },
    "Vision-Language Models": {
        "zen-vl-4b-instruct": {
            "hf_id": "zenlm/zen-vl-4b-instruct",
            "type": "vision-language",
            "size": "4B",
            "context": "32K"
        },
        "zen-vl-8b-instruct": {
            "hf_id": "zenlm/zen-vl-8b-instruct",
            "type": "vision-language",
            "size": "8B",
            "context": "32K"
        },
        "zen-vl-30b-instruct": {
            "hf_id": "zenlm/zen-vl-30b-instruct",
            "type": "vision-language",
            "size": "30B",
            "context": "32K"
        },
    }
}

# Dataset configurations
DATASETS = {
    "Agent Training": {
        "ADP - AgentTuning OS": {
            "hf_id": "neulab/agent-data-collection",
            "config": "agenttuning_os",
            "size": "~5k samples"
        },
        "ADP - AgentTuning KG": {
            "hf_id": "neulab/agent-data-collection",
            "config": "agenttuning_kg",
            "size": "~5k samples"
        },
        "ADP - AgentTuning DB": {
            "hf_id": "neulab/agent-data-collection",
            "config": "agenttuning_db",
            "size": "~5k samples"
        },
        "ADP - Synatra": {
            "hf_id": "neulab/agent-data-collection",
            "config": "synatra",
            "size": "99k samples"
        },
        "ADP - Code Feedback": {
            "hf_id": "neulab/agent-data-collection",
            "config": "code_feedback",
            "size": "66k samples"
        },
        "ADP - Go Browse": {
            "hf_id": "neulab/agent-data-collection",
            "config": "go-browse-wa",
            "size": "27k samples"
        },
    },
    "Function Calling": {
        "xLAM Function Calling 60k": {
            "hf_id": "Salesforce/xlam-function-calling-60k",
            "config": None,
            "size": "60k samples"
        },
    },
    "Instruction Tuning": {
        "Alpaca": {
            "hf_id": "tatsu-lab/alpaca",
            "config": None,
            "size": "52k samples"
        },
    }
}

def train_model(
    model_name: str,
    selected_datasets: List[str],
    max_samples: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_repo: str
):
    """Main training function"""
    
    try:
        logs = []
        
        def log(msg):
            print(msg)
            logs.append(msg)
            yield "\n".join(logs)
        
        yield from log("=" * 80)
        yield from log("🧘 ZEN TRAINING SPACE")
        yield from log("=" * 80)
        yield from log("")
        
        # GPU info
        yield from log(f"🎮 GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            yield from log(f"   Device: {torch.cuda.get_device_name(0)}")
            yield from log(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        yield from log("")
        
        # Find model config
        model_config = None
        for category in MODELS.values():
            if model_name in category:
                model_config = category[model_name]
                break
        
        if not model_config:
            yield from log(f"❌ Model {model_name} not found")
            return
        
        yield from log(f"📦 Loading model: {model_name}")
        yield from log(f"   HF ID: {model_config['hf_id']}")
        yield from log(f"   Size: {model_config['size']}")
        yield from log(f"   Type: {model_config['type']}")
        
        # Load model
        model = AutoModel.from_pretrained(
            model_config['hf_id'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if model_config['type'] == "vision-language":
            processor = AutoProcessor.from_pretrained(model_config['hf_id'])
        else:
            processor = AutoTokenizer.from_pretrained(model_config['hf_id'])
        
        yield from log("✅ Model loaded")
        yield from log("")
        
        # Load datasets
        yield from log("📚 Loading datasets...")
        all_datasets = []
        
        for dataset_name in selected_datasets:
            # Find dataset config
            dataset_config = None
            for category in DATASETS.values():
                if dataset_name in category:
                    dataset_config = category[dataset_name]
                    break
            
            if not dataset_config:
                yield from log(f"⚠️  Dataset {dataset_name} not found, skipping")
                continue
            
            yield from log(f"   Loading: {dataset_name}")
            yield from log(f"   HF ID: {dataset_config['hf_id']}")
            
            try:
                if dataset_config['config']:
                    ds = load_dataset(
                        dataset_config['hf_id'],
                        dataset_config['config'],
                        split="train",
                        streaming=True
                    )
                else:
                    ds = load_dataset(
                        dataset_config['hf_id'],
                        split="train",
                        streaming=True
                    )
                
                # Take limited samples
                samples = []
                for i, example in enumerate(ds):
                    if i >= max_samples // len(selected_datasets):
                        break
                    samples.append(example)
                
                all_datasets.extend(samples)
                yield from log(f"   ✅ Loaded {len(samples)} samples")
                
            except Exception as e:
                yield from log(f"   ❌ Error: {e}")
        
        yield from log(f"\n✅ Total samples loaded: {len(all_datasets)}")
        yield from log("")
        
        # Training setup
        yield from log("⚙️  Training Configuration:")
        yield from log(f"   Epochs: {epochs}")
        yield from log(f"   Batch Size: {batch_size}")
        yield from log(f"   Learning Rate: {learning_rate}")
        yield from log(f"   Samples: {len(all_datasets)}")
        yield from log(f"   Output: {output_repo}")
        yield from log("")
        
        training_args = TrainingArguments(
            output_dir="./training-output",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=100,
            bf16=True,
            push_to_hub=True,
            hub_model_id=output_repo,
            report_to="tensorboard",
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=all_datasets if len(all_datasets) > 0 else None,
        )
        
        # Train!
        yield from log("🔥 TRAINING STARTED")
        yield from log("=" * 80)
        
        result = trainer.train()
        
        yield from log("")
        yield from log("=" * 80)
        yield from log("✅ TRAINING COMPLETED!")
        yield from log("=" * 80)
        yield from log(f"📊 Final Loss: {result.training_loss:.4f}")
        yield from log(f"☁️  Model uploaded to: {output_repo}")
        yield from log("")
        yield from log("🎉 SUCCESS!")
        
    except Exception as e:
        yield from log(f"\n❌ ERROR: {str(e)}")
        import traceback
        yield from log(f"\n{traceback.format_exc()}")

# Build Gradio Interface
with gr.Blocks(title="Zen Training Space", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧘 Zen Training Space
    ### Unified Training Platform for All Zen Models
    
    Train any Zen model with any dataset combination from HuggingFace.
    All datasets are loaded directly from HF - no local storage needed!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Select Model")
            
            model_choice = gr.Dropdown(
                choices=[
                    *[f"{cat} / {model}" for cat in MODELS for model in MODELS[cat]]
                ],
                label="Model",
                value="Vision-Language Models / zen-vl-4b-instruct"
            )
            
            gr.Markdown("### 2. Select Datasets")
            
            dataset_choices = gr.CheckboxGroup(
                choices=[
                    *[f"{cat} / {ds}" for cat in DATASETS for ds in DATASETS[cat]]
                ],
                label="Datasets",
                value=[
                    "Agent Training / ADP - Synatra",
                    "Function Calling / xLAM Function Calling 60k"
                ]
            )
            
            gr.Markdown("### 3. Training Config")
            
            max_samples = gr.Slider(100, 100000, value=10000, step=100, label="Max Samples")
            epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
            batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch Size")
            learning_rate = gr.Number(value=2e-5, label="Learning Rate")
            
            output_repo = gr.Textbox(
                value="zenlm/zen-vl-4b-agent-custom",
                label="Output Repository (HuggingFace)"
            )
            
            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### Training Logs")
            output = gr.Textbox(label="", lines=35, max_lines=50, show_label=False)
    
    train_btn.click(
        train_model,
        inputs=[
            model_choice,
            dataset_choices,
            max_samples,
            epochs,
            batch_size,
            learning_rate,
            output_repo
        ],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### 📊 Available Models
    - **Language**: nano (0.6B), eco (4B), omni (7B), coder (14B), next (32B)
    - **Vision-Language**: zen-vl (4B, 8B, 30B)
    
    ### 📚 Available Datasets
    - **Agent Training**: ADP (220k+ trajectories across 15+ configs)
    - **Function Calling**: xLAM (60k high-quality examples)
    - **Instruction**: Alpaca (52k samples)
    
    ### 💰 Cost Estimates (HF Pro GPU)
    - 4B model: $3-5 for 10k samples
    - 8B model: $8-12 for 10k samples
    - 32B model: $30-50 for 10k samples
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
