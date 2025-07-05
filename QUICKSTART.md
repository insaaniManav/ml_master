# ML-Master Quick Start Guide

This guide will help you get started with the ML-Master framework in minutes.

## 🚀 Quick Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ml_master
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Azure OpenAI credentials:**
   ```bash
   export AZURE_OPENAI_API_KEY="your-api-key-here"
   export AZURE_OPENAI_ENDPOINT="https://your-endpoint.cognitiveservices.azure.com/"
   export AZURE_OPENAI_MODEL="o3"
   export AZURE_OPENAI_DEPLOYMENT="o3"
   ```

## 🎯 Your First ML-Master Task

### Option 1: Command Line Interface

```bash
python main.py --task "Create a binary classification model for the iris dataset using scikit-learn"
```

### Option 2: Python Script

```python
from core.integration import MLMaster
import yaml

# Load configuration
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize ML-Master
ml_master = MLMaster(config)

# Define your task
task_description = """
Create a machine learning solution for binary classification.
Requirements:
1. Load and preprocess the data
2. Split into training/testing sets
3. Train a classification model
4. Evaluate with accuracy, precision, recall, F1-score
5. Print results
"""

# Solve the task
results = ml_master.solve_task(
    task_description=task_description,
    max_time=300,  # 5 minutes
    max_iterations=50
)

# View results
print(f"Best Score: {results['solution']['performance_score']}")
print(f"Generated Code:\n{results['solution']['code']}")
```

## 📊 Understanding the Results

ML-Master provides comprehensive results including:

- **Solution**: Generated code and performance score
- **Performance**: Iterations, time, and efficiency metrics
- **Exploration**: Tree statistics and search behavior
- **Memory**: Usage statistics and insights captured
- **Framework**: Version and configuration info

## ⚙️ Configuration Options

### Key Configuration Parameters

```yaml
# Exploration settings
exploration:
  max_iterations: 1000      # MCTS iterations
  num_workers: 3           # Parallel workers
  uct_constant: 1.414      # Exploration-exploitation balance

# Reasoning settings
reasoning:
  model_name: "o3"         # Azure OpenAI model
  max_context_length: 8000 # Context window size
  enable_self_verification: true

# Memory settings
memory:
  max_memory_size: 1000    # Memory entries limit
  context_window_size: 10  # Recent context window
```

### Environment Variables

```bash
# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_MODEL="o3"
export AZURE_OPENAI_DEPLOYMENT="o3"

# Framework settings
export ML_MASTER_LOG_LEVEL="INFO"
export ML_MASTER_MAX_ITERATIONS="1000"
export ML_MASTER_MAX_TIME="3600"
```

## 🔧 Advanced Usage

### Custom Task Types

ML-Master can handle various ML tasks:

```bash
# Classification
python main.py --task "Create a neural network for MNIST digit classification"

# Regression
python main.py --task "Build a regression model for house price prediction"

# Clustering
python main.py --task "Implement K-means clustering for customer segmentation"

# Time Series
python main.py --task "Create an LSTM model for stock price prediction"
```

### Performance Tuning

```bash
# More iterations for complex tasks
python main.py --task "Your task" --max-iterations 2000

# Longer time limit
python main.py --task "Your task" --max-time 1800

# Debug mode for detailed logging
python main.py --debug --task "Your task"
```

### Custom Configuration

```bash
# Use custom config file
python main.py --config my_config.yaml --task "Your task"
```

## 📈 Monitoring and Debugging

### Log Files

Logs are automatically saved to:
- `logs/ml_master.log` - Main application logs
- `logs/ml_master_example.log` - Example execution logs

### Framework Statistics

```python
# Get comprehensive stats
stats = ml_master.get_framework_stats()
print(f"Memory usage: {stats['memory']['memory_usage_percent']}%")
print(f"Total nodes explored: {stats['exploration']['total_nodes']}")
```

### Node Information

```python
# Get details about specific exploration nodes
node_info = ml_master.get_node_info("node_id_here")
print(f"Node depth: {node_info['depth']}")
print(f"Performance score: {node_info['solution']['performance_score']}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Azure OpenAI API Error**
   ```
   Error: AZURE_OPENAI_API_KEY environment variable is required
   ```
   **Solution**: Set your API key: `export AZURE_OPENAI_API_KEY="your-key"`

2. **Memory Issues**
   ```
   Error: Memory limit exceeded
   ```
   **Solution**: Reduce `max_memory_size` in config or increase system memory

3. **Timeout Errors**
   ```
   Error: Code execution timed out
   ```
   **Solution**: Increase `execution_timeout` in config or simplify the task

4. **Import Errors**
   ```
   Error: Module not found
   ```
   **Solution**: Install missing dependencies: `pip install -r requirements.txt`

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
python main.py --debug --task "Your task"
```

## 📚 Next Steps

1. **Read the full documentation** in the README.md
2. **Explore examples** in the `examples/` directory
3. **Customize configuration** for your specific needs
4. **Contribute** to the project on GitHub

## 🆘 Getting Help

- **Documentation**: Check the README.md for detailed information
- **Issues**: Report bugs on GitHub Issues
- **Examples**: Look at the `examples/` directory for more use cases
- **Configuration**: Review `configs/config.yaml` for all options

---

**Happy AI-for-AI development with ML-Master! 🚀** 