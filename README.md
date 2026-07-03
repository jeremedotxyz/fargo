[README.md](https://github.com/user-attachments/files/29626659/README.md)
# llm-tool-harness

A minimal, extensible tool-calling agent harness for OpenAI models. Drop it into a repo, add your own tools, and you have a working agent loop with retries, memory, and logging in under a hundred lines of "core" code.

## Why this exists

Most agent frameworks bury the actual loop under abstraction. This harness keeps the loop visible and hackable:

```
user message -> model -> (tool calls? -> execute -> feed results back -> model) -> final answer
```

Everything else (tools, memory, config) is a plain Python object you can swap out.

## Structure

```
llm-tool-harness/
├── src/harness/
│   ├── agent.py          # The core agent loop
│   ├── config.py         # Env-based configuration
│   ├── llm_client.py     # Thin OpenAI wrapper
│   ├── memory.py         # Conversation memory (in-memory + optional JSON persistence)
│   └── tools/
│       ├── base.py       # Tool base class / schema
│       ├── registry.py   # Registers tools, builds OpenAI tool schemas, dispatches calls
│       └── builtin.py    # Example tools (calculator, clock, file reader)
├── examples/
│   └── run_cli.py        # Interactive REPL entry point
├── tests/
│   └── test_agent.py
├── .env.example
├── requirements.txt
└── LICENSE
```

## Setup

```bash
git clone <this-repo>
cd llm-tool-harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

## Run it

```bash
python examples/run_cli.py
```

```
> what's 2847 * 193, and what time is it right now?
[tool call] calculator(expression="2847 * 193")
[tool call] get_current_time()
2847 * 193 = 549,471. Current time: 2026-07-03T18:42:01Z
```

## Adding a tool

Tools are plain functions wrapped in a `Tool` object. Add one in `src/harness/tools/builtin.py` (or your own module) and register it:

```python
from harness.tools.base import Tool

def get_weather(city: str) -> str:
    return f"Sunny in {city}, 72F"  # replace with a real API call

weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
    function=get_weather,
)
```

Then register it with the agent:

```python
from harness.tools.registry import ToolRegistry
from harness.agent import Agent

registry = ToolRegistry()
registry.register(weather_tool)

agent = Agent(tool_registry=registry)
agent.run("What's the weather in Vancouver, WA?")
```

## Config

All config lives in `.env` / environment variables, read via `src/harness/config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `HARNESS_MODEL` | `gpt-4.1` | model name |
| `HARNESS_MAX_ITERATIONS` | `8` | tool-call loop safety cap |
| `HARNESS_TEMPERATURE` | `0.7` | sampling temperature |
| `HARNESS_MEMORY_PATH` | *(unset)* | if set, conversation is persisted to this JSON file |

## Testing

```bash
pytest tests/
```

## Notes

- The loop caps at `HARNESS_MAX_ITERATIONS` tool round-trips per call to `run()` to avoid runaway loops — tune this per use case.
- `memory.py` is intentionally dumb (a list of messages). Swap in a vector store or summarizer once you outgrow it.
- No hidden network calls except to `api.openai.com` and whatever your own tools call out to.

## License

MIT
