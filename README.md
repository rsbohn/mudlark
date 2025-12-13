# MUD Hub

A three-way text-routing hub enabling conversation between User, MUD, and LLM.

## Installation

This project uses `uv` for dependency management.

```bash
uv sync
```

## Usage

### With MUD Connection

```bash
uv run main.py <mud_host> <mud_port>
```

Or use the script entry point:
```bash
uv run mudhub <mud_host> <mud_port>
```

Example:
```bash
uv run main.py aardmud.org 23
```

### Local Mode (LLM only, no MUD)

```bash
uv run main.py --local
```

Or:
```bash
uv run mudhub --local
```

In local mode, all user input (except hub commands) is sent directly to the LLM.

### Quick Fake MUD for Testing

You can spin up a tiny fake MUD using `socat` and `fake_mud.py`:

```bash
# In one terminal (requires socat installed)
chmod +x fake_mud.py
socat -v TCP-LISTEN:4000,reuseaddr,fork EXEC:"./fake_mud.py"
# If socat can't find python on PATH, use: EXEC:"/usr/bin/env python3 fake_mud.py"

# In another terminal, run mudhub against it
uv run main.py localhost 4000
```

Sending movement commands like `n/s/e/w` will rotate between a couple of room descriptions so you can test location detection, quiet/trace toggles, and LLM proposals without a live MUD.

## User Interface

### Input Routing
- `// <message>` - Send to LLM
- `[command]` - Hub command
- `<anything else>` - Send directly to MUD (or LLM in --local mode)

### Hub Commands
- `[help]` - Show available commands
- `[proposals]` - List pending LLM proposals
- `[approve N]` - Send proposal #N to MUD
- `[reject N]` - Discard proposal #N
- `[llm on|off]` - Toggle LLM observation

## Architecture

The hub maintains three streams:
- **MUD ↔ Hub**: Telnet connection (via telnetlib3)
- **User ↔ Hub**: stdin/stdout
- **LLM ↔ Hub**: API calls (to be implemented)

## LLM Integration

LLM integration is stubbed. To complete:
1. Add your LLM API client (OpenAI, Anthropic, etc.)
2. Implement `llm_observe()` to send context
3. Implement `llm_query()` to make API calls
4. Parse responses for `SAY:` and `NOTE:` prefixes

## Safety Model

- LLM has zero direct write access to MUD
- All LLM messages require explicit approval
- Prevents loops, spam, runaway automation

## Design Philosophy

Policy lives at the edges, not the center. This system favors:
- Explicit control
- Transparent data flow
- Composability
- Debuggability
