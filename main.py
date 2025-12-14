#!/usr/bin/env python3
"""
Mudlark (Codex)
A three-way text-routing hub enabling conversation between User, MUD, and LLM.
"""

import asyncio
import sys
import os
import csv
import argparse
from datetime import datetime
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
        self.mudlist: Dict[str, Dict[str, str]] = {}
        self.current_location: Optional[str] = None
        self.location_block: List[str] = []  # holds recent lines to detect room changes
        self.location_task: Optional[asyncio.Task] = None
        self.location_llm_delay: float = 0.6
        self.awaiting_room_header: bool = True
        self.llm_quiet: bool = False
        self.llm_trace: bool = False
        self.llm_auto: bool = False
        
        self.running = False
        
        # Load MUD list
        self.load_mudlist()
        
        # Initialize LLM model
        try:
            self.model = llm.get_model(self.llm_model)
        except Exception as e:
            print(f"[HUB] Warning: Could not load LLM model '{self.llm_model}': {e}", file=sys.stderr)
            self.model = None
            self.llm_enabled = False
    
    def load_mudlist(self):
        """Load MUD list from mudlist.csv."""
        try:
            with open('mudlist.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row['name'].strip()
                    self.mudlist[name] = {
                        'host': row['address'].strip(),
                        'port': int(row['port'].strip())
                    }
        except FileNotFoundError:
            print("[HUB] Warning: mudlist.csv not found", file=sys.stderr)
        except Exception as e:
            print(f"[HUB] Warning: Error loading mudlist.csv: {e}", file=sys.stderr)
    
    async def connect_mud(self, host: Optional[str] = None, port: Optional[int] = None):
        """Establish telnet connection to MUD."""
        # Use provided host/port or fall back to instance variables
        connect_host = host or self.mud_host
        connect_port = port or self.mud_port
        
        try:
            self.mud_reader, self.mud_writer = await open_connection(
                connect_host, connect_port
            )
            self.mud_host = connect_host
            self.mud_port = connect_port
            self.log_transcript(f"[HUB] Connected to {connect_host}:{connect_port}")
            return True
        except Exception as e:
            print(f"[HUB ERROR] Failed to connect to MUD: {e}", file=sys.stderr)
            return False
    
    def log_transcript(self, message: str):
        """Add message to transcript buffer."""
        self.transcript.append(message)
    
    def append_to_bambu(self, line: str):
        """Append a line to bambu.log, best-effort."""
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("bambu.log", "a") as f:
                f.write(f"[{ts}] {line}\n")
        except Exception as e:
            print(f"[HUB] Failed to write to bambu.log: {e}", file=sys.stderr)

    def log_session_start(self):
        """Log a session boundary to bambu.log."""
        self.append_to_bambu("=== session start ===")
    
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

                # Detect location changes and trigger LLM automatically
                self.process_location_detection(text)
                    
        except Exception as e:
            print(f"\n[HUB ERROR] MUD reader error: {e}", file=sys.stderr)
        finally:
            # Clean up connection but keep hub running
            if self.mud_writer:
                self.mud_writer.close()
                try:
                    await self.mud_writer.wait_closed()
                except:
                    pass
            self.mud_reader = None
            self.mud_writer = None
    
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
        
        elif command.startswith('[dial '):
            mud_name = command[6:-1].strip() if command.endswith(']') else command[6:].strip()
            await self.dial_mud(mud_name)
        
        elif command.startswith('[note'):
            note_text = command[5:]
            if note_text.endswith(']'):
                note_text = note_text[:-1]
            note_text = note_text.strip()
            if not note_text:
                print("[HUB] Usage: [note <text>]", file=sys.stderr)
                return
            self.append_to_bambu(f"[note]{note_text}")
            print("[HUB] Note appended to bambu.log")
        
        elif command.startswith('[llm '):
            state = command[5:-1].strip() if command.endswith(']') else command[5:].strip()
            if state == 'on':
                self.llm_enabled = True
                print("[HUB] LLM observation enabled")
            elif state == 'off':
                self.llm_enabled = False
                print("[HUB] LLM observation disabled")
            elif state == 'quiet on':
                self.llm_quiet = True
                print("[HUB] LLM quiet mode enabled (auto room queries suppressed)")
            elif state == 'quiet off':
                self.llm_quiet = False
                print("[HUB] LLM quiet mode disabled (auto room queries resume)")
            elif state == 'trace on':
                self.llm_trace = True
                print("[HUB] LLM trace enabled (will print prompts sent to LLM)")
            elif state == 'trace off':
                self.llm_trace = False
                print("[HUB] LLM trace disabled")
            elif state == 'auto on':
                self.llm_auto = True
                print("[HUB] LLM auto-execute enabled (proposals execute automatically)")
            elif state == 'auto off':
                self.llm_auto = False
                print("[HUB] LLM auto-execute disabled (proposals require approval)")
            elif state == 'info':
                self.show_llm_info()
            else:
                print("[HUB] Usage: [llm on|off|quiet on|quiet off|trace on|trace off|auto on|auto off|info]", file=sys.stderr)
        
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
  [note <text>]  - Append <text> to bambu.log
  [dial <name>]  - Connect to MUD from mudlist.csv
  [llm on|off|quiet on|quiet off|trace on|trace off|auto on|auto off|info] - Toggle LLM settings
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
        Proposal._next_id = 1
        print(f"[HUB] Rejected all {count} proposal(s)")
    
    async def dial_mud(self, mud_name: str):
        """Connect to a MUD from the mudlist."""
        # Check if already connected
        if self.mud_writer and not self.local_mode:
            print("[HUB] Already connected. Disconnect first.")
            return
        
        # Look up MUD in mudlist
        if mud_name not in self.mudlist:
            print(f"[HUB] MUD '{mud_name}' not found in mudlist.csv", file=sys.stderr)
            return
        
        mud_info = self.mudlist[mud_name]
        host = mud_info['host']
        port = mud_info['port']
        
        print(f"[HUB] Dialing {mud_name} at {host}:{port}...")
        
        # If in local mode, exit it and start MUD connection
        if self.local_mode:
            self.local_mode = False
        
        if await self.connect_mud(host, port):
            # Start the MUD reader loop
            asyncio.create_task(self.mud_reader_loop())
    
    def show_llm_info(self):
        """Display LLM model and status information."""
        model_obj = type(self.model).__name__ if self.model else "None"
        info_line = f"LLM Info | model={self.llm_model} loaded={self.model is not None} enabled={self.llm_enabled} quiet={self.llm_quiet} trace={self.llm_trace} auto={self.llm_auto} pending={len(self.proposals)} obj={model_obj}"
        print(f"[HUB] {info_line}")
        self.append_to_bambu(info_line)

    def process_location_detection(self, text: str):
        """Heuristically detect room changes from MUD output and auto-ping the LLM."""
        for raw_line in text.splitlines():
            line = raw_line.rstrip('\r')
            if self.awaiting_room_header:
                stripped = line.strip()
                if self.is_prompt_line(stripped) or stripped.startswith('<'):
                    # Ignore prompts while waiting for the next header
                    self.location_block.clear()
                    self.awaiting_room_header = True
                elif self.is_room_header(line):
                    self.location_block = [line]
                    self.awaiting_room_header = False
                continue

            # Accumulate lines until we see the "Obvious exits" marker
            self.location_block.append(line)
            if len(self.location_block) > 80:
                self.location_block = self.location_block[-80:]

            if 'obvious exits' in line.lower():
                room_name = self.extract_room_name(self.location_block)
                self.location_block.clear()
                self.awaiting_room_header = True

                if room_name and room_name != self.current_location:
                    self.current_location = room_name
                    self.append_to_bambu(f"[location]{room_name}")
                    if not self.llm_quiet:
                        print(f"[HUB] New location detected: {room_name} (LLM in {self.location_llm_delay:.1f}s)")
                        self.schedule_location_llm(room_name)

    def extract_room_name(self, lines: List[str]) -> Optional[str]:
        """Best-effort extraction of the room name from a description block."""
        if not lines:
            return None
        for line in lines:
            lower = line.lower()
            if 'obvious exits' in lower:
                break
            if self.is_prompt_line(line):
                continue
            candidate = line.strip()
            if self.is_room_header(candidate):
                return candidate
        return None

    def is_room_header(self, line: str) -> bool:
        """Check if a line looks like a room name."""
        stripped = line.strip()
        if not stripped:
            return False
        if self.is_prompt_line(stripped):
            return False
        if '<' in stripped or '>' in stripped:
            return False
        if line[:1].isspace():
            return False
        if stripped.endswith('.'):
            return False
        if ' says ' in stripped.lower():
            return False
        # Heuristic: room headers usually don't contain ':' (exit lines do)
        if ':' in stripped:
            return False
        if stripped.lower() in {'n', 's', 'e', 'w', 'u', 'd', 'ne', 'nw', 'se', 'sw', 'in', 'out', 'look', 'l'}:
            return False
        return True

    def is_prompt_line(self, line: str) -> bool:
        """Detect common MUD prompt lines so we don't mistake them for room headers."""
        lower = line.lower()
        if '<' in line and '>' in line and ('hp' in lower or 'mv' in lower or 'm ' in lower):
            return True
        if lower.startswith('<') and ('hp' in lower or 'mv' in lower):
            return True
        if lower.endswith('>') and ('hp' in lower or 'mv' in lower):
            return True
        return False

    def schedule_location_llm(self, room_name: str):
        """Start (or restart) delayed LLM query after movement."""
        # Respect user choice if LLM is disabled or unavailable
        if not self.llm_enabled or not self.model or self.llm_quiet:
            return

        if self.location_task and not self.location_task.done():
            self.location_task.cancel()
        self.location_task = asyncio.create_task(self._delayed_location_llm(room_name))

    async def _delayed_location_llm(self, room_name: str):
        try:
            await asyncio.sleep(self.location_llm_delay)
            await self.llm_query(f"New location: {room_name}. Provide concise next steps based on recent context.")
        except asyncio.CancelledError:
            return
        finally:
            self.location_task = None
    
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
        
        if not self.llm_quiet:
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

            if self.llm_trace:
                print("[HUB] --- LLM TRACE BEGIN ---")
                print(full_prompt)
                print("[HUB] --- LLM TRACE END ---")
            
            # Run LLM query in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.prompt(full_prompt).text()
            )
            
            # Parse the response
            await self.parse_llm_response(response)
            
        except Exception as e:
            print(f"[HUB] LLM error: {e}", file=sys.stderr)
    
    async def parse_llm_response(self, response: str):
        """Parse LLM response and extract proposals."""
        lines = response.split('\n')
        existing = {p.text.strip().lower() for p in self.proposals.values()}
        directions = {"n", "s", "e", "w", "u", "d", "ne", "nw", "se", "sw",
                      "north", "south", "east", "west", "up", "down", "in", "out"}
        
        for line in lines:
            line = line.strip()
            if line.startswith('SAY:'):
                text = line[4:].strip()
                norm = text.lower()
                if norm in directions:
                    continue
                if norm.startswith("go "):
                    maybe_dir = norm[3:].strip()
                    if maybe_dir in directions:
                        continue
                if norm in existing:
                    continue
                existing.add(norm)
                proposal = Proposal(text)
                self.proposals[proposal.id] = proposal
                
                if self.llm_auto:
                    # Auto-execute the proposal
                    await self.send_to_mud(proposal.text)
                    self.log_transcript(f"[LLM->MUD] {proposal.text}")
                    print(f"[HUB] Auto-executed {proposal}")
                    del self.proposals[proposal.id]
                else:
                    print(f"[HUB] New proposal {proposal}")
            elif line.startswith('NOTE:'):
                note = line[5:].strip()
                if not self.llm_quiet:
                    print(f"[LLM] {note}")
            elif line:
                # Treat unprefixed lines as NOTE
                if not self.llm_quiet:
                    print(f"[LLM] {line}")
    
    def get_context(self, max_lines: int = 50) -> str:
        """Get recent transcript for LLM context."""
        # Get only the most recent messages to avoid stale context
        recent = list(self.transcript)[-max_lines:]
        return '\n'.join(recent)
    
    async def run(self):
        """Main event loop."""
        self.running = True
        
        if self.local_mode:
            print("[HUB] Local mode - LLM only. Type [help] for commands.")
        else:
            if await self.connect_mud():
                # Start MUD reader loop as background task
                asyncio.create_task(self.mud_reader_loop())
            print("[HUB] Mudlark started. Type [help] for commands.")
        
        # Mark start of session in log
        self.log_session_start()
        
        try:
            # User input loop runs until explicit quit
            await self.user_input_loop()
        except KeyboardInterrupt:
            print("\n[HUB] Shutting down...")
        finally:
            if self.mud_writer:
                self.mud_writer.close()
                try:
                    await self.mud_writer.wait_closed()
                except:
                    pass
    
    async def shutdown(self):
        """Clean shutdown."""
        self.running = False


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Mudlark - MUD/LLM Hub')
    parser.add_argument('host', nargs='?', help='MUD host or @mudname')
    parser.add_argument('port', nargs='?', type=int, help='MUD port')
    parser.add_argument('--local', action='store_true', help='Start in local mode (LLM only)')
    parser.add_argument('--llm_model', default='gpt-4o-mini', help='LLM model to use (default: gpt-4o-mini)')
    
    args = parser.parse_args()
    
    # Check for --local flag
    if args.local:
        hub = MUDHub(local_mode=True, llm_model=args.llm_model)
        asyncio.run(hub.run())
        return
    
    if not args.host:
        parser.print_help()
        sys.exit(1)
    
    # Check if first argument is @mudname format
    if args.host.startswith('@'):
        mud_name = args.host[1:]  # Remove @ prefix
        hub = MUDHub(local_mode=True, llm_model=args.llm_model)  # Start in local mode
        # Load the mudlist to validate
        if mud_name not in hub.mudlist:
            print(f"Error: MUD '{mud_name}' not found in mudlist.csv")
            sys.exit(1)
        # Set the initial MUD to dial
        mud_info = hub.mudlist[mud_name]
        hub.mud_host = mud_info['host']
        hub.mud_port = mud_info['port']
        hub.local_mode = False
        asyncio.run(hub.run())
        return
    
    if not args.port:
        print("Error: Port required when not using @mudname format")
        parser.print_help()
        sys.exit(1)
    
    hub = MUDHub(args.host, args.port, llm_model=args.llm_model)
    asyncio.run(hub.run())


if __name__ == '__main__':
    main()
