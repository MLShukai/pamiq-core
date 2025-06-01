# VAE Torch Sample

This sample demonstrates how to build a **Variational Autoencoder (VAE)** using PyTorch and the `pamiq_core` framework. It showcases the framework's modular approach to machine learning experiments by implementing a complete VAE training pipeline with agent-environment interactions, model management, and data handling.

## What This Sample Teaches

- **Modular ML Architecture**: How to structure ML experiments using `pamiq_core`'s component-based design
- **Agent-Environment Pattern**: How to implement the agent-environment paradigm for VAE training
- **Model Management**: How to wrap PyTorch models for training and inference
- **Data Flow**: How to manage data buffers and streaming in ML pipelines
- **Training Orchestration**: How to coordinate multiple components in a training loop

______________________________________________________________________

## Quick Start

```bash
# Clone repository
git clone https://github.com/MLShukai/pamiq-core

# Install dependencies
cd pamiq-core/pamiq-core/samples/vae-torch
uv sync

# Run the VAE training
uv run python main.py
```

> **Note:** If you don't have `uv` installed, follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

The sample will train a VAE on synthetic data (zero tensors) and output training logs. Training automatically uses GPU if available.

### Stopping Training

After sufficient time has passed, you can stop the training using several methods and then view the results with TensorBoard:

**Method 1: Keyboard Interrupt (Simple)**

```bash
# Press Ctrl+C in the terminal where main.py is running
```

**Method 2: Remote Console (Recommended)**

```bash
# In a separate terminal, connect to the running system
pamiq-console --host localhost --port 8391

# Then use the shutdown command
pamiq-console (active) > shutdown
```

### Viewing Results

```bash
# View training metrics and logs
tensorboard --logdir=runs
```

The TensorBoard interface will be available at `http://localhost:6006`.

![resulting_tensorboard](./docs/images/resulting_tensorboard.png)

______________________________________________________________________

## Project Architecture

```
vae-torch/
├── main.py                # 🚀 Entry point and orchestration
├── agent.py               # 🤖 EncodingAgent implementation
├── env.py                 # 🌍 Environment for data provision
├── model.py               # 🧠 VAE neural network architecture
└── trainer.py             # 🏋️ Training logic and loss computation
```

## How to Read This Project

**Start here for different learning goals:**

### 🎯 **Understanding the Overall Flow**

- **`main.py`** - Entry point that orchestrates all components
    - Shows how to configure device selection
    - Demonstrates component registration and launch

### 🤖 **Agent-Environment Pattern**

- **`agent.py`** - Implements the encoding agent
    - Returns latent representations as "actions"
    - Shows how to access inference models
- **`env.py`** - Provides training data as "observations"
    - Generates synthetic data (zero tensors)
    - Validates agent actions for consistency

### 🧠 **Model Architecture**

- **`model.py`** - VAE implementation (Encoder + Decoder)
    - 3-layer encoder with mean/logvar outputs
    - 3-layer decoder for reconstruction
    - Includes reparameterization trick
    - Custom `Encoder.infer` method: We can configure inference flow independently from training (`Encoder.forward`), by passing the method `Encoder.infer` into `inference_procedure` attribute when constructing `TorchTrainingModel` (in `main.py`)

### 🏋️ **Training Logic**

- **`trainer.py`** - VAE-specific training procedures
    - Implements VAE loss (reconstruction + KL divergence)
    - Handles optimizer setup and training loops
    - Includes TensorBoard logging

### 📊 **Data Management**

- **Data buffer setup in `main.py`**
    - Shows `RandomReplacementBuffer` usage
    - Demonstrates data flow configuration

## Expected Output

When you run the sample, you'll see:

1. Console logs
2. TensorBoard logs
3. Model state saving to `./states/` directory

## References

- [PyTorch Documentation](https://pytorch.org/)
- [pamiq-core Documentation](https://mlshukai.github.io/pamiq-core/)
