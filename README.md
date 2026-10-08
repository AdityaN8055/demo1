"""Interactive work agent with permission-controlled workspace file access.

Install: pip install openai
Set OPENAI_API_KEY, then run: python prog1.py
The agent's workspace is the current directory. File writes require approval.
"""

import json
import os
from pathlib import Path

from openai import OpenAI


WORKSPACE = Path.cwd().resolve()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TOOLS = [
	{
		"type": "function",
		"function": {
			"name": "list_files",
			"description": "List files and folders inside the workspace.",
			"parameters": {
				"type": "object",
				"properties": {"folder": {"type": "string", "description": "Relative folder path; defaults to the workspace root."}},
				"required": [],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "read_file",
			"description": "Read a UTF-8 text file inside the workspace.",
			"parameters": {
				"type": "object",
				"properties": {"path": {"type": "string", "description": "Path relative to the workspace."}},
				"required": ["path"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "write_file",
			"description": "Create or replace a UTF-8 text file. Always ask the user before writing.",
			"parameters": {
				"type": "object",
				"properties": {
					"path": {"type": "string", "description": "Path relative to the workspace."},
					"content": {"type": "string"},
				},
				"required": ["path", "content"],
			},
		},
	},
]


def safe_path(relative_path):
	"""Resolve a path and prevent access outside the workspace."""
	path = (WORKSPACE / relative_path).resolve()
	if path != WORKSPACE and WORKSPACE not in path.parents:
		raise ValueError("Path must remain inside the workspace.")
	return path


def run_tool(name, arguments):
	try:
		if name == "list_files":
			folder = safe_path(arguments.get("folder", "."))
			if not folder.is_dir():
				return "Folder not found."
			entries = sorted(item.name + ("/" if item.is_dir() else "") for item in folder.iterdir())
			return "\n".join(entries) if entries else "Folder is empty."

		if name == "read_file":
			path = safe_path(arguments["path"])
			return path.read_text(encoding="utf-8")[:20000]

		if name == "write_file":
			path = safe_path(arguments["path"])
			print(f"Agent requests writing {path.relative_to(WORKSPACE)}")
			if input("Approve this write? [y/N] ").strip().lower() != "y":
				return "Write declined by the user."
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_text(arguments["content"], encoding="utf-8")
			return f"Successfully wrote {path.relative_to(WORKSPACE)}."

		return "Unknown tool."
	except (OSError, ValueError, KeyError) as error:
		return f"Tool error: {error}"


def main():
	client = OpenAI()
	messages = [{
		"role": "system",
		"content": (
			"You are a practical work assistant. Help the user plan and complete tasks. "
			"Use file tools when helpful, only access the workspace, and never claim an action "
			"succeeded unless its tool result confirms it. Be concise and ask before consequential actions."
		),
	}]

	print(f"Work agent ready. Workspace: {WORKSPACE}")
	print("Enter a task, or type 'exit' to quit.")
	while True:
		task = input("\nYou: ").strip()
		if task.lower() in {"exit", "quit"}:
			break
		if not task:
			continue
		messages.append({"role": "user", "content": task})

		for _ in range(8):
			response = client.chat.completions.create(
				model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto"
			)
			reply = response.choices[0].message
			messages.append(reply.model_dump(exclude_none=True))
			if not reply.tool_calls:
				print(f"\nAgent: {reply.content or ''}")
				break

			for call in reply.tool_calls:
				try:
					arguments = json.loads(call.function.arguments)
					result = run_tool(call.function.name, arguments)
				except (json.JSONDecodeError, TypeError) as error:
					result = f"Invalid tool arguments: {error}"
				messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
		else:
			print("\nAgent: Stopped after the tool-call limit. Try breaking the task into smaller steps.")


if __name__ == "__main__":
	main()
