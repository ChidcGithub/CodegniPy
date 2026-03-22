# CogniPy

[![PyPI version](https://img.shields.io/pypi/v/cognipy.svg)](https://pypi.org/project/cognipy/)
[![Python](https://img.shields.io/pypi/pyversions/cognipy.svg)](https://pypi.org/project/cognipy/)
[![License](https://img.shields.io/github/license/ChidcGithub/CogniPy)](LICENSE)
[![Build Status](https://github.com/ChidcGithub/CogniPy/actions/workflows/ci.yml/badge.svg)](https://github.com/ChidcGithub/CogniPy/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/codecov/c/github/ChidcGithub/CogniPy.svg)](https://codecov.io/gh/ChidcGithub/CogniPy)
[![Code Quality](https://img.shields.io/lgtm/grade/python/g/ChidcGithub/CogniPy.svg)](https://lgtm.com/projects/g/ChidcGithub/CogniPy/)

**AI-Native Python Language Extension**

CogniPy seamlessly integrates Large Language Models (LLMs) into Python, making non-deterministic AI capabilities a first-class citizen in your code. Write logic close to natural language while achieving production-grade performance and debuggability.

## Features

- **Syntactic Extension**: `~` operator for natural language prompts directly in code
- **Cognitive Decorator**: `@cognitive` decorator to let LLMs implement functions
- **Memory Management**: Session-level memory with pluggable storage backends
- **Reflection Loop**: Built-in self-correction and quality assurance
- **Async Scheduler**: High-performance concurrent LLM calls with priority queuing
- **Deterministic Guarantees**: Type constraints, simulation mode, and hallucination detection

## Installation

```bash
pip install cognipy
```

For development:

```bash
pip install cognipy[dev]
```

## Quick Start

### The `~` Operator

```python
from cognipy import CognitiveContext, cognitive_call

with CognitiveContext(api_key="your-api-key"):
    # Natural language prompts directly in code
    result = ~"Translate to English: Hello World"
    print(result)
```

### The `@cognitive` Decorator

```python
from cognipy import cognitive, CognitiveContext

@cognitive
def summarize(text: str) -> str:
    """Summarize the key points of this text in no more than two sentences."""

with CognitiveContext(api_key="your-api-key"):
    summary = summarize("Python is a high-level programming language...")
    print(summary)
```

### Async Batch Processing

```python
import asyncio
from cognipy import batch_call, CognitiveContext

async def main():
    prompts = [
        "Translate: Hello",
        "Translate: World", 
        "Translate: Python"
    ]
    results = await batch_call(prompts, max_concurrent=3)
    print(results)

asyncio.run(main())
```

### With Memory Persistence

```python
from cognipy import CognitiveContext, FileStore

with CognitiveContext(
    api_key="your-api-key",
    memory_store=FileStore("session_memory.json")
):
    cognitive_call("My name is Alice")
    response = cognitive_call("What is my name?")
    # LLM will remember: "Alice"
```

### With Type Constraints

```python
from cognipy import PrimitiveConstraint, deterministic_call

# Ensure LLM output is a valid integer between 0-100
constraint = PrimitiveConstraint(
    int,
    min_value=0,
    max_value=100
)

result = deterministic_call(
    "Generate a random number between 1 and 100",
    constraint
)

if result.status == "valid":
    print(result.value)  # Guaranteed valid integer
```

### With Reflection

```python
from cognipy import CognitiveContext, with_reflection

with CognitiveContext(api_key="your-api-key") as ctx:
    result = with_reflection(
        "Explain quantum entanglement",
        context=ctx,
        max_iterations=2
    )
    
    if result.status == "passed":
        print(result.corrected_response or result.original_response)
```

## Architecture

```
Python Source Code
        |
        v
  AST Preprocessing
        |
        v
Transformed Code + cognitive_call()
        |
        v
   Runtime Layer
        |
        v
  Scheduler (async)
        |
        v
    LLM APIs
        |
        v
  Validation Layer
        |
        v
 Deterministic Result
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `cognitive_call(prompt, context=None, model=None, temperature=None)` | Execute a cognitive call to LLM |
| `deterministic_call(prompt, constraint, context=None)` | Call LLM with type constraints |
| `batch_call(prompts, max_concurrent=5)` | Execute multiple prompts concurrently |

### Decorators

| Decorator | Description |
|-----------|-------------|
| `@cognitive` | Decorate a function to be implemented by LLM |
| `@cognitive(model="gpt-4")` | With specific model selection |

### Context Manager

```python
CognitiveContext(
    api_key=None,           # OpenAI API key (or use OPENAI_API_KEY env var)
    model="gpt-4o-mini",    # Default model
    base_url=None,          # Custom API endpoint
    temperature=0.7,        # Sampling temperature
    max_tokens=1024,        # Maximum response tokens
    memory_store=None       # Memory storage backend
)
```

### Memory Backends

| Class | Description |
|-------|-------------|
| `InMemoryStore` | Volatile in-memory storage |
| `FileStore(path)` | Persistent file-based storage |

### Type Constraints

| Class | Description |
|-------|-------------|
| `PrimitiveConstraint(type, min_value=None, max_value=None, min_length=None, max_length=None, pattern=None)` | Validate primitive types |
| `EnumConstraint(values)` | Validate enum values |
| `SchemaConstraint(pydantic_model)` | Validate against Pydantic schema |
| `ListConstraint(item_constraint, min_items=None, max_items=None)` | Validate list items |

### Scheduler

```python
from cognipy import CognitiveScheduler, Priority

scheduler = CognitiveScheduler(
    max_concurrent=5,
    default_timeout=30.0,
    retry_policy=RetryPolicy(max_retries=3, base_delay=1.0)
)

# Submit with priority
task_id = await scheduler.submit(
    my_coroutine,
    priority=Priority.HIGH,
    timeout=60.0
)

# Get result
result = await scheduler.get_result(task_id, timeout=10.0)
```

### Hallucination Detection

```python
from cognipy import HallucinationDetector

detector = HallucinationDetector()
check = detector.detect(llm_response)

print(check.confidence)  # 0.0 - 1.0
print(check.issues)      # List of detected issues
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `COGNIPY_MODEL` | Default model to use |
| `COGNIPY_TEMPERATURE` | Default temperature |
| `COGNIPY_MAX_TOKENS` | Default max tokens |

### Programmatic Configuration

```python
from cognipy import CognitiveContext

ctx = CognitiveContext(
    api_key="sk-...",
    model="gpt-4",
    temperature=0.5,
    max_tokens=2048
)
```

## CLI Usage

```bash
# Run a script with cognitive features
cognipy run script.py

# Start interactive REPL
cognipy repl

# Show version
cognipy version

# With options
cognipy run script.py --model gpt-4 --api-key sk-...
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=cognipy --cov-report=html
```

### Simulation Mode

For testing without actual LLM calls:

```python
from cognipy import Simulator, SimulationMode

simulator = Simulator(mode=SimulationMode.MOCK)

# Mock responses
simulator.add_mock("Hello", "Hi there!")

# Or record real responses for replay
simulator = Simulator(mode=SimulationMode.RECORD)
# ... make real calls ...
simulator.save_recordings("recordings.json")

# Later, replay them
simulator = Simulator(mode=SimulationMode.REPLAY)
simulator.load_recordings("recordings.json")
```

## Project Structure

```
CogniPy/
  cognipy/
    __init__.py          # Package exports
    runtime.py           # Core runtime (cognitive_call, CognitiveContext)
    decorator.py         # @cognitive decorator
    transformer.py       # AST transformer for ~ operator
    memory.py            # Memory storage backends
    reflection.py        # Reflection loop implementation
    scheduler.py         # Async scheduler with retry/timeout
    determinism.py       # Type constraints, simulator, hallucination detection
    cli.py               # Command-line interface
  tests/
    test_transformer.py
    test_memory.py
    test_scheduler.py
    test_determinism.py
  examples/
    demo.py
  pyproject.toml
  README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/ChidcGithub/CogniPy.git
cd CogniPy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check cognipy/
mypy cognipy/
```

## Roadmap

- [ ] Rust extension for high-performance scheduling
- [ ] Support for more LLM providers (Anthropic, local models)
- [ ] Enhanced hallucination detection with external verification
- [ ] Visual debugging tools
- [ ] Distributed execution support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

CogniPy is inspired by the vision of making AI a natural part of programming, bridging the gap between deterministic code and probabilistic intelligence.

## Links

- [Documentation](https://github.com/ChidcGithub/CogniPy#readme)
- [Issue Tracker](https://github.com/ChidcGithub/CogniPy/issues)
- [PyPI Package](https://pypi.org/project/cognipy/)
