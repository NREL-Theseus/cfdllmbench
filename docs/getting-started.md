# Getting Started

This guide will help you set up and run CFDLLMBench to evaluate Large Language Models on CFD tasks.

## Prerequisites

- Python 3.8 or higher
- Git
- OpenFOAM (for FoamBench evaluation)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/NREL-Theseus/cfdllmbench.git
cd cfdllmbench
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up OpenFOAM (Optional)

For FoamBench evaluation, install OpenFOAM:

```bash
# Ubuntu/Debian
sudo apt-get install openfoam

# Or follow official OpenFOAM installation guide
```

## Quick Start

### Running CFDQuery

Evaluate conceptual understanding with multiple-choice questions:

```bash
python run_cfdquery.py --model gpt-3.5-turbo --output results/
```

### Running CFDCodeBench

Test code generation capabilities:

```bash
python run_cfdcodebench.py --model gpt-3.5-turbo --output results/
```

### Running FoamBench

Assess end-to-end OpenFOAM automation:

```bash
python run_foambench.py --model gpt-3.5-turbo --level basic --output results/
```

## Configuration

Configure your LLM API keys and settings in `config.yaml`:

```yaml
models:
  openai:
    api_key: "your-api-key-here"
    model: "gpt-3.5-turbo"
  
evaluation:
  timeout: 300
  max_retries: 3
```

## Next Steps

- [Explore Benchmark Components](components/overview.md)
- [View Results and Analysis](results.md)
- [Check API Reference](api.md)
