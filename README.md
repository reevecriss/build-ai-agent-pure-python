# Build AI Agent in Pure Python

A lightweight, robust, and educational guide and framework for building autonomous AI agents from scratch using **pure Python**, without heavy framework abstractions.

---

## Table of Contents

- [Overview](#overview)
- [Core Architecture of an AI Agent](#core-architecture-of-an-ai-agent)
- [Project Structure](#project-structure)
- [Implementation Guide](#implementation-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Defining Tools](#2-defining-tools)
  - [3. The ReAct Agent Loop](#3-the-react-agent-loop)
  - [4. Putting It All Together](#4-putting-it-all-together)
- [Running the Agent](#running-the-agent)
- [Advanced Extensions](#advanced-extensions)

---

## Overview

Modern agent frameworks often introduce layers of abstraction that obscure how agents actually work under the hood. This guide demonstrates how to build a fully functional **ReAct (Reason + Act)** AI agent using vanilla Python, direct LLM API calls, and standard library primitives.

---

## Core Architecture of an AI Agent

An autonomous agent operates in a continuous feedback loop:

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│ Perception│ ──> │  Reason   │ ──> │  Action   │ ──> │ Observation│
└───────────┘     └───────────┘     └───────────┘     └───────────┘
      ▲                                                      │
      └──────────────────────────────────────────────────────┘
```

1. **Perceive:** Receive user prompt or environment state.
2. **Reason (Thought):** LLM analyzes the context and decides whether to respond directly or call a tool.
3. **Act (Tool Execution):** Execute the chosen Python function with arguments provided by the LLM.
4. **Observation:** Feed the tool execution output back into the agent's context loop.

---

## Project Structure

```
ai_agent_pure_python/
├── agent.py       # Core agent loop and LLM integration
├── tools.py       # Custom Python functions exposed to the agent
├── prompts.py     # System prompts and ReAct instructions
├── requirements.txt # Minimal dependencies (e.g., requests or openai)
└── main.py        # Entry point
```

---

## Implementation Guide

### 1. Environment Setup

Create `requirements.txt`:
```txt
requests>=2.28.0
python-dotenv>=1.0.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Defining Tools (`tools.py`)

Tools are ordinary Python functions with type hints and docstrings. The LLM relies heavily on docstrings to understand what a tool does and what arguments it expects.

```python
import json
import urllib.request
import datetime

def get_current_time(timezone: str = "UTC") -> str:
    """Returns the current date and time."""
    now = datetime.datetime.now()
    return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"

def calculate(expression: str) -> str:
    """Safely evaluates a mathematical expression."""
    try:
        # Restricted evaluation for safety in production
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

# Registry mapping tool names to functions
TOOL_REGISTRY = {
    "get_current_time": get_current_time,
    "calculate": calculate
}
```

### 3. The ReAct Agent Loop (`agent.py`)

```python
import os
import json
import requests
from tools import TOOL_REGISTRY

class PurePythonAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are an autonomous AI assistant capable of using tools. "
                    "When you need to use a tool, output JSON in the format: "
                    '{"thought": "...", "tool": "tool_name", "arguments": {...}}.\n'
                    "When you have the final answer, output: "
                    '{"thought": "...", "answer": "..."}'
                )
            }
        ]

    def call_llm(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def run(self, user_prompt: str, max_steps: int = 5):
        self.messages.append({"role": "user", "content": user_prompt})
        
        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")
            response_text = self.call_llm(self.messages)
            print(f"LLM Response:\n{response_text}")
            
            try:
                # Parse JSON response from LLM
                response_json = json.loads(response_text)
            except json.JSONDecodeError:
                print("Error: LLM did not return valid JSON. Retrying...")
                self.messages.append({"role": "user", "content": "Please respond in valid JSON format as instructed."})
                continue

            # Check if agent finished
            if "answer" in response_json:
                print(f"\nFinal Answer: {response_json['answer']}")
                return response_json['answer']

            # Execute Tool
            tool_name = response_json.get("tool")
            tool_args = response_json.get("arguments", {})

            if tool_name in TOOL_REGISTRY:
                print(f"Executing tool '{tool_name}' with args {tool_args}...")
                tool_func = TOOL_REGISTRY[tool_name]
                try:
                    observation = tool_func(**tool_args)
                except Exception as e:
                    observation = f"Tool execution failed: {e}"
                
                print(f"Observation: {observation}")
                
                # Append interaction history
                self.messages.append({"role": "assistant", "content": response_text})
                self.messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                error_msg = f"Error: Tool '{tool_name}' not found."
                print(error_msg)
                self.messages.append({"role": "user", "content": error_msg})

        print("Max steps reached without final answer.")
        return None
```

### 4. Entry Point (`main.py`)

```python
import os
from agent import PurePythonAgent

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable.")
        return

    agent = PurePythonAgent(api_key=api_key)
    
    prompt = "What is 1452 * 342, and what is the current UTC time?"
    agent.run(prompt)

if __name__ == "__main__":
    main()
```

---

## Running the Agent

1. Set your API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
2. Run the main script:
   ```bash
   python main.py
   ```

---

## Advanced Extensions

- **Memory Persistence:** Store conversation logs in SQLite or JSON files.
- **Async Execution:** Use `asyncio` and `httpx` for asynchronous tool calls and parallel agent tasks.
- **Human-in-the-Loop:** Intercept dangerous tool calls (e.g., file deletion, database writes) to prompt for user confirmation.
