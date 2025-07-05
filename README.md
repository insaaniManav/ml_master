# ML Master: AI-for-AI Framework

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.16499-b31b1b.svg)](https://arxiv.org/abs/2506.16499)
[![MLE-Bench](https://img.shields.io/badge/MLE--Bench-29.3%25%20Medal%20Rate-brightgreen.svg)](https://github.com/openai/mle-bench)

ML Master is a novel AI-for-AI (AI4AI) framework that seamlessly integrates exploration and reasoning to automate various AI tasks and generate different types of content. This framework achieves state-of-the-art performance on MLE-Bench with a 29.3% average medal rate and extends beyond code generation to support markdown, plain text, and structured data generation.

**Steerable Reasoning**: Unlike traditional GPT that generates one-shot solutions, ML Master uses advanced reasoning that can be guided and controlled. It breaks down complex problems into manageable steps, validates its own work, and adapts its approach based on feedback from previous attempts.

**Adaptive Memory System**: ML Master remembers insights from every attempt and uses this knowledge to improve future solutions. It selectively captures relevant information, learns from failures, and builds upon successful strategies - essentially learning and improving with each task it solves.

**How it works**: ML Master explores multiple solution paths simultaneously using Monte Carlo Tree Search (MCTS), remembers what works and what doesn't, and continuously refines its approach. The result? Solutions that are not just generated, but evolved through intelligent exploration and learning.

## Research Foundation

This project is based on the research paper **"ML-Master: Towards AI-for-AI via Integration of Exploration and Reasoning"** (arXiv:2506.16499v1). The framework implements the paper's key innovation: seamless integration of exploration and reasoning through adaptive memory mechanisms.

> **Paper**: [ML-Master: Towards AI-for-AI via Integration of Exploration and Reasoning](https://arxiv.org/abs/2506.16499)

## Why ML Master is Better Than Regular GPT

### Traditional GPT Limitations
- **Single-shot generation**: GPT generates one solution without exploration
- **No memory between attempts**: Each generation starts from scratch
- **Limited reasoning depth**: No systematic exploration of solution space
- **No self-improvement**: Cannot learn from previous attempts

### ML Master Advantages
- **Multi-trajectory exploration**: Uses MCTS to explore multiple solution paths
- **Adaptive memory system**: Remembers and leverages insights from previous attempts
- **Steerable reasoning**: Enhanced reasoning with memory integration
- **Self-verification**: Validates and improves its own solutions
- **Parallel execution**: Explores multiple solutions simultaneously

## Key Features

### Multi-Content Type Generation
- **Code Generation**: Python, JavaScript, Java, SQL, HTML/CSS with syntax validation
- **Markdown Generation**: Documentation, READMEs, guides with proper formatting
- **Plain Text Generation**: Explanations, reports, summaries with clear structure
- **Structured Data Generation**: JSON, YAML, XML configurations with schema validation

### Intelligent Content Type Detection
- Automatic detection of content type from task description
- Keyword-based classification for optimal generation
- Manual override capability for specific requirements

### Advanced Reasoning & Exploration
- MCTS-inspired tree search for optimal solution exploration
- Steerable reasoning with adaptive memory integration
- Parallel execution across multiple workers
- Self-verification and validation mechanisms 
## Architecture

```
ML-Master/
├── main.py                 # Main entry point and CLI interface
├── core/                   # Core framework components
│   ├── exploration/        # Multi-trajectory exploration module
│   ├── reasoning/         # Steerable reasoning module
│   ├── memory/            # Adaptive memory mechanism
│   └── integration/       # Framework integration
├── configs/               # Configuration files
├── utils/                 # Utility functions
└── README.md             # This file
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ml_master

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_MODEL="your-model-name"
```

### Conversational Bot (Recommended)

The easiest way to interact with ML-Master is through the conversational bot:

```bash
# Start the interactive chat bot
python chat_bot.py

# Start with custom settings
python chat_bot.py --user-name "Alice" --debug

# Run demo to see examples
python demo_chat.py
```

The conversational bot provides:
- Natural language interaction
- Automatic content type detection
- Conversation history
- Framework statistics
- Special commands for different tasks

### Command Line Interface

For programmatic usage, use the main.py script:

```bash
# Generate code
python main.py --generate --task "Create a Python function to sort a list" --content-type code

# Generate documentation
python main.py --generate --task "Write a README for a machine learning project" --content-type markdown

# Auto-detect content type
python main.py --generate --task "Write a Python script to process data"

# Full task solving with exploration
python main.py --task "Implement a machine learning pipeline for image classification"
```

### Programmatic Usage

For more advanced usage, you can use the MLMaster class directly:

```python
from core.integration import MLMaster
from core.integration.content_types import ContentType
import yaml

# Load configuration
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize framework
ml_master = MLMaster(config)

# Generate content
content_output = ml_master.generate_content(
    task_description="Create a Python function for data preprocessing",
    content_type=ContentType.CODE
)

print(f"Generated {content_output.content_type.value}:")
print(content_output.content)
```

### Main.py - Your Starting Point

The `main.py` file serves as both the entry point and a comprehensive example. It demonstrates:
- Command-line interface usage
- All content type generation modes
- Full task solving with exploration
- Configuration management
- Error handling and logging

Run `python main.py --help` to see all available options.

## Content Type Examples

### Code Generation
```python
def factorial(n):
    """Calculate factorial of a number with error handling."""
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 0:
        raise ValueError("Input must be non-negative")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
```

### Markdown Generation
```markdown
# Machine Learning Project

## Installation
```bash
pip install -r requirements.txt
```
## Technical Stack

- **Language**: Python 3.9+
- **LLM Integration**: Azure OpenAI
- **Content Processing**: Syntax validation, formatting, schema validation
- **Parallel Processing**: Multiprocessing, AsyncIO
- **Data Storage**: SQLite, JSON
- **Testing**: pytest, unittest
- **Monitoring**: Logging, Metrics collection

## Success Criteria

1. **Performance**: Achieve >25% medal rate on MLE-Bench
2. **Efficiency**: Complete tasks within 12-hour time limit
3. **Reliability**: >90% valid submission rate

## Configuration

### Environment Variables
```bash
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_ENDPOINT="your-endpoint"
AZURE_OPENAI_MODEL="o3"
AZURE_OPENAI_DEPLOYMENT="o3"
ML_MASTER_LOG_LEVEL="INFO"
ML_MASTER_MAX_ITERATIONS="1000"
```

### Key Parameters
- **Exploration**: MCTS iterations, workers, UCT constant
- **Reasoning**: Azure OpenAI settings, context management
- **Memory**: Size limits, decay factors, compression
- **Execution**: Timeouts, resource limits, security



## Monitoring & Debugging

### Logging
- **Console Output**: Real-time progress and results
- **File Logs**: Detailed logs in `logs/` directory
- **Debug Mode**: `--debug` flag for verbose output

### Statistics
- **Framework Stats**: Memory usage, exploration metrics
- **Performance Metrics**: Iterations, time, efficiency
- **Node Information**: Detailed node exploration data

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use ML Master in your research, please cite the original paper:

```bibtex
@misc{liu2025mlmasteraiforaiintegrationexploration,
      title={ML-Master: Towards AI-for-AI via Integration of Exploration and Reasoning}, 
      author={Zexi Liu and Yuzhu Cai and Xinyu Zhu and Yujie Zheng and Runkun Chen and Ying Wen and Yanfeng Wang and Weinan E and Siheng Chen},
      year={2025},
      eprint={2506.16499},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.16499}, 
}
```

## Support

For questions, issues, or contributions, please:
1. Check the existing issues
2. Create a new issue with detailed information
3. Join our community discussions



## Acknowledgments

This project is based on research from the paper "ML-Master: Towards AI-for-AI via Integration of Exploration and Reasoning" and builds upon the MLE-Bench evaluation framework. 