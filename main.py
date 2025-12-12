#!/usr/bin/env python3
"""
Three-Way MUD Hub (Codex)
A text-routing hub enabling conversation between User, MUD, and LLM.
"""

import asyncio
import sys
import os
from typing import Optional, List, Dict
from collections import deque
from telnetlib3 import open_connection
import llm


class Proposal:
    """A candidate message from the LLM."""
    _next_id = 1
    
    def __init__(self, text: str):
        self.id = Proposal._next_id
        Proposal._next_id += 1
        self.text = text
    
    def __str__(self):
        return f"#{self.id}: {self.text}"


class MUDHub:
    """Central hub managing three-way communication."""
    
    def __init__(self, mud_host: Optional[str] = None, mud_port: Optional[int] = None, llm_enabled: bool = True, local_mode: bool = False, llm_model: str = "gpt-4o-mini"):
        self.mud_host = mud_host
        self.mud_port = mud_port
        self.llm_enabled = llm_enabled
        self.local_mode = local_mode
        self.llm_model = llm_model
        
        self.mud_reader: Optional[asyncio.StreamReader] = None
        self.mud_writer: Optional[asyncio.StreamWriter] = None
        
        self.proposals: Dict[int, Proposal] = {}
        self.transcript: deque = deque(maxlen=1000)
        
        self.running = False
        
        # Initialize LLM model
        try:
            self.model = llm.get_model(self.llm_model)
        except Exception as e:
            print(f"[HUB] Warning: Could not load LLM model '{self.llm_model}': {e}", file=sys.stderr)
            self.model = None
            self.llm_enabled = False
    
    async def connect_mud(self):
        """Establish telnet connection to MUD."""
        try:
            self.mud_reader, self.mud_writer = await open_connection(
                self.mud_host, self.mud_port
            )
            self.log_transcript(f"[HUB] Connected to {self.mud_host}:{self.mud_port}")
            return True
        except Exception as e:
            print(f"[HUB ERROR] Failed to connect to MUD: {e}", file=sys.stderr)
            return False
    
    def log_transcript(self, message: str):
        """Add message to transcript buffer."""
        self.transcript.append(message)
    
    async def mud_reader_loop(self):
        """Read from MUD and display to user + send to LLM."""
        try:
            while self.running and self.mud_reader:
                data = await self.mud_reader.read(4096)
                if not data:
                    print("\n[HUB] MUD connection closed", file=sys.stderr)
                    break
                
                text = data
                # Display raw to user
                sys.stdout.write(text)
                sys.stdout.flush()
                
                # Log to transcript
                self.log_transcript(f"[MUD] {text}")
                
                # TODO: Send to LLM for observation if enabled
                if self.llm_enabled:
                    await self.llm_observe(text, "mud")
                    
        except Exception as e:
            print(f"\n[HUB ERROR] MUD reader error: {e}", file=sys.stderr)
        finally:
            self.running = False
    
    async def user_input_loop(self):
        """Read from stdin and route based on prefix."""
        loop = asyncio.get_event_loop()
        
        try:
            while self.running:
                # Read line from stdin (non-blocking)
                line = await loop.run_in_executor(None, sys.stdin.readline)
                
                if not line:  # EOF
                    break
                
                line = line.rstrip('\n\r')
                
                if not line:
                    continue
                
                # Route based on prefix
                if line.startswith('#'):
                    # Shortcut to approve proposal
                    try:
                        prop_id = int(line[1:].strip())
                        await self.approve_proposal(prop_id)
                    except ValueError:
                        # Not a valid #N pattern, send to MUD
                        if self.local_mode:
                            self.log_transcript(f"[USER->LLM] {line}")
                            await self.llm_query(line)
                        else:
                            await self.send_to_mud(line)
                            self.log_transcript(f"[USER->MUD] {line}")
                    
                elif line.startswith('//'):
                    # LLM command
                    prompt = line[2:].strip()
                    self.log_transcript(f"[USER->LLM] {prompt}")
                    await self.llm_query(prompt)
                    
                elif line.startswith('['):
                    # Hub command
                    await self.handle_hub_command(line)
                    
                else:
                    # Send to MUD (or in local mode, treat as LLM query)
                    if self.local_mode:
                        self.log_transcript(f"[USER->LLM] {line}")
                        await self.llm_query(line)
                    else:
                        await self.send_to_mud(line)
                        self.log_transcript(f"[USER->MUD] {line}")
                    
        except Exception as e:
            print(f"\n[HUB ERROR] User input error: {e}", file=sys.stderr)
        finally:
            self.running = False
    
    async def send_to_mud(self, message: str):
        """Send a message to the MUD."""
        if self.mud_writer:
            self.mud_writer.write(message + '\r\n')
            await self.mud_writer.drain()
    
    async def handle_hub_command(self, command: str):
        """Process hub commands."""
        command = command.strip()
        
        if command == '[help]':
            self.show_help()
        
        elif command == '[proposals]':
            self.show_proposals()
        
        elif command.startswith('[approve '):
            parts = command[9:-1].strip() if command.endswith(']') else command[9:].strip()
            try:
                prop_id = int(parts)
                await self.approve_proposal(prop_id)
            except ValueError:
                print("[HUB] Invalid proposal ID", file=sys.stderr)
        
        elif command.startswith('[reject '):
            parts = command[8:-1].strip() if command.endswith(']') else command[8:].strip()
            if parts == 'all':
                self.reject_all_proposals()
            else:
                try:
                    prop_id = int(parts)
                    self.reject_proposal(prop_id)
                except ValueError:
                    print("[HUB] Invalid proposal ID", file=sys.stderr)
        
        elif command.startswith('[llm '):
            state = command[5:-1].strip() if command.endswith(']') else command[5:].strip()
            if state == 'on':
                self.llm_enabled = True
                print("[HUB] LLM observation enabled")
            elif state == 'off':
                self.llm_enabled = False
                print("[HUB] LLM observation disabled")
            elif state == 'info':
                self.show_llm_info()
            else:
                print("[HUB] Usage: [llm on|off|info]", file=sys.stderr)
        
        elif command in ('[quit]', '[q]', '[exit]'):
            print("[HUB] Shutting down...")
            self.running = False
        
        else:
            print(f"[HUB] Unknown command: {command}", file=sys.stderr)
            self.show_help()
    
    def show_help(self):
        """Display available hub commands."""
        help_text = """
[HUB] Available commands:
  [help]         - Show this help
  [proposals]    - List queued proposals
  [approve N]    - Send proposal N to the MUD
  [reject N]     - Discard proposal N
  [reject all]   - Discard all proposals
  [llm on|off|info] - Toggle LLM observation or show LLM info
  [quit]         - Exit the hub (also: [q], [exit])
  
User input routing:
  #N             - Approve proposal N (shortcut)
  //             - Send message to LLM
  [command]      - Hub command
  <anything>     - Send directly to MUD
"""
        print(help_text)
    
    def show_proposals(self):
        """Display all pending proposals."""
        if not self.proposals:
            print("[HUB] No pending proposals")
        else:
            print("[HUB] Pending proposals:")
            for prop in self.proposals.values():
                print(f"  {prop}")
    
    async def approve_proposal(self, prop_id: int):
        """Approve and send a proposal to the MUD."""
        if prop_id not in self.proposals:
            print(f"[HUB] Proposal #{prop_id} not found", file=sys.stderr)
            return
        
        proposal = self.proposals[prop_id]
        await self.send_to_mud(proposal.text)
        self.log_transcript(f"[LLM->MUD] {proposal.text}")
        print(f"[HUB] Approved #{prop_id}: {proposal.text}")
        del self.proposals[prop_id]
    
    def reject_proposal(self, prop_id: int):
        """Reject and discard a proposal."""
        if prop_id not in self.proposals:
            print(f"[HUB] Proposal #{prop_id} not found", file=sys.stderr)
            return
        
        proposal = self.proposals[prop_id]
        print(f"[HUB] Rejected #{prop_id}: {proposal.text}")
        del self.proposals[prop_id]
    
    def reject_all_proposals(self):
        """Reject and discard all proposals."""
        if not self.proposals:
            print("[HUB] No proposals to reject")
            return
        
        count = len(self.proposals)
        self.proposals.clear()
        print(f"[HUB] Rejected all {count} proposal(s)")
    
    def show_llm_info(self):
        """Display LLM model and status information."""
        print("[HUB] LLM Info:")
        print(f"  Model: {self.llm_model}")
        print(f"  Enabled: {self.llm_enabled}")
        print(f"  Loaded: {self.model is not None}")
        if self.model:
            print(f"  Model object: {type(self.model).__name__}")
        print(f"  Pending proposals: {len(self.proposals)}")
    
    async def llm_observe(self, text: str, source: str):
        """Send observation to LLM."""
        if not self.model:
            return
        
        # For now, just log the observation to transcript
        # In future, could send observations to build context for the LLM
        self.log_transcript(f"[{source.upper()}] {text}")
    
    async def llm_query(self, prompt: str):
        """Query the LLM and parse response into proposals."""
        if not self.llm_enabled:
            print("[HUB] LLM is disabled")
            return
        
        if not self.model:
            print("[HUB] LLM model not loaded")
            return
        
        print(f"[HUB] LLM query: {prompt}")
        
        try:
            # Build context from recent transcript
            context = self.get_context()
            
            # Create system prompt
            system_prompt = """You are an AI assistant helping a user interact with a MUD (Multi-User Dungeon) game.
When you want to suggest commands to send to the MUD, prefix them with "SAY:" on their own line.
For commentary or explanation, prefix lines with "NOTE:" or leave them unprefixed.

Example response:
NOTE: I see you're in a tavern. Let me look around.
SAY: look
SAY: inventory

Only suggest MUD commands, never execute them directly."""
            
            # Combine context and prompt
            full_prompt = f"{system_prompt}\n\nRecent context:\n{context}\n\nUser query: {prompt}"
            
            # Run LLM query in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.prompt(full_prompt).text()
            )
            
            # Parse the response
            self.parse_llm_response(response)
            
        except Exception as e:
            print(f"[HUB] LLM error: {e}", file=sys.stderr)
    
    def parse_llm_response(self, response: str):
        """Parse LLM response and extract proposals."""
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('SAY:'):
                text = line[4:].strip()
                proposal = Proposal(text)
                self.proposals[proposal.id] = proposal
                print(f"[HUB] New proposal {proposal}")
            elif line.startswith('NOTE:'):
                note = line[5:].strip()
                print(f"[LLM] {note}")
            elif line:
                # Treat unprefixed lines as NOTE
                print(f"[LLM] {line}")
    
    def get_context(self) -> str:
        """Get recent transcript for LLM context."""
        return '\n'.join(self.transcript)
    
    async def run(self):
        """Main event loop."""
        if self.local_mode:
            print("[HUB] Local mode - LLM only. Type [help] for commands.")
            self.running = True
        else:
            if not await self.connect_mud():
                return
            self.running = True
            print("[HUB] MUD Hub started. Type [help] for commands.")
        
        try:
            # Run loops concurrently (only mud_reader_loop if not in local mode)
            if self.local_mode:
                await self.user_input_loop()
            else:
                await asyncio.gather(
                    self.mud_reader_loop(),
                    self.user_input_loop()
                )
        except KeyboardInterrupt:
            print("\n[HUB] Shutting down...")
        finally:
            if self.mud_writer:
                self.mud_writer.close()
                await self.mud_writer.wait_closed()
    
    async def shutdown(self):
        """Clean shutdown."""
        self.running = False


def main():
    """Entry point."""
    # Check for --local flag
    if '--local' in sys.argv:
        hub = MUDHub(local_mode=True)
        asyncio.run(hub.run())
        return
    
    if len(sys.argv) < 3:
        print("Usage: uv run main.py <mud_host> <mud_port>")
        print("   or: uv run main.py --local")
        sys.exit(1)
    
    mud_host = sys.argv[1]
    try:
        mud_port = int(sys.argv[2])
    except ValueError:
        print("Error: Port must be a number")
        sys.exit(1)
    
    hub = MUDHub(mud_host, mud_port)
    asyncio.run(hub.run())


if __name__ == '__main__':
    main()
