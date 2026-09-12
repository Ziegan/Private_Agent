import os
import sys
import time
import asyncio
from typing import List
from rich.console import Console
from rich.panel import Panel

import ollama
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama

from .config import (
    CONFIG_FILE_PATH,
    DEFAULT_DB_PATH,
    WORKSPACE_ROOT_DEFAULT,
    SKILLS_FOLDER_DEFAULT,
    RAG_DOCS_DEFAULT,
    MODEL_TEMPERATURE
)
from .database import PersistentMemory
from .sandbox import SandboxManager
from .tools import AVAILABLE_TOOLS, set_active_db_path
from .rag import initialize_knowledge_base
from .skills import load_skills_from_folder, match_skill_by_relevancy

console = Console()

def check_hardware_backend() -> str:
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        if "GPU" in devices:
            return "GPU Accelerated"
        return "(OpenVINO Runtime)"
    except Exception:
        return "Standard Ollama Backend"

def fetch_local_chat_models() -> List[str]:
    try:
        response = ollama.list()
        models_list = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        chat_models = []
        embedding_keywords = ["embed", "embedding", "bge", "e5", "nomic-embed"]

        for m in models_list:
            model_name = m.get("model", "") if isinstance(m, dict) else getattr(m, "model", "")
            if not model_name:
                continue
            if any(kw in model_name.lower() for kw in embedding_keywords):
                continue
            chat_models.append(model_name)
        return chat_models
    except Exception as e:
        console.print(f"[red][Warning] Could not connect to Ollama: {e}[/red]")
        return []

def estimate_context_window(chat_history: list, current_input: str, max_context: int = 32768) -> dict:
    total_chars = len(current_input)
    for msg in chat_history:
        content = msg.content
        content_str = "".join([str(c) for c in content]) if isinstance(content, list) else str(content)
        total_chars += len(content_str)
    
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        estimated_tokens = len(encoding.encode(current_input + "".join([str(m.content) for m in chat_history])))
    except Exception:
        estimated_tokens = int(total_chars / 3.5)

    percentage = min(100.0, (estimated_tokens / max_context) * 100)
    return {"tokens": estimated_tokens, "max": max_context, "percent": round(percentage, 2)}

async def execute_tool_call(tool_call: dict) -> ToolMessage:
    name, args, call_id = tool_call.get("name"), tool_call.get("args", {}), tool_call.get("id")
    console.print(f"[yellow][Tool Execution][/yellow] Calling '{name}' with validated args: {args}")

    tool_func = AVAILABLE_TOOLS.get(name)
    if tool_func:
        try:
            if hasattr(tool_func, "ainvoke"):
                result = await tool_func.ainvoke(args)
            else:
                result = tool_func.invoke(args)
        except Exception as e:
            result = f"Error executing tool {name}: {str(e)}"
    else:
        result = f"Error: Tool {name} not found."

    if "error" in str(result).lower() or "exit code: 1" in str(result).lower() or "traceback" in str(result).lower():
        console.print(f"[yellow][Reflection Loop] Tool execution error detected. Feeding trace back to model for auto-debugging...[/yellow]")
        result += "\n[System Reflection Prompt]: Your execution encountered an error/traceback. Analyze why it failed, correct your approach, and try again."

    return ToolMessage(content=str(result), tool_call_id=call_id)

async def run_agent_cli_async():
    # Single unified big welcome banner
    console.print(Panel("[bold cyan]Private Agent (mcp & async)[/bold cyan]", title="Welcome", border_style="cyan"))

    hardware_info = check_hardware_backend()
    console.print(f"[bold green][Hardware Status][/bold green] Backend Target: {hardware_info}")
    console.print(f"[bold green][Config Status][/bold green] Loaded configuration from: [cyan]{CONFIG_FILE_PATH.resolve()}[/cyan]")

    memory = PersistentMemory(db_path=DEFAULT_DB_PATH)
    set_active_db_path(DEFAULT_DB_PATH)

    docs_input = RAG_DOCS_DEFAULT if RAG_DOCS_DEFAULT else console.input("[yellow]Enter knowledge base (KB) files directory path (Press Enter to skip): [/yellow]").strip()
    vectorstore = initialize_knowledge_base(docs_input if docs_input else None)
    is_rag_active = vectorstore is not None

    skills_folder_input = SKILLS_FOLDER_DEFAULT if SKILLS_FOLDER_DEFAULT else console.input("[cyan]Enter path to skills folder containing .md files (Press Enter for none): [/cyan]").strip()
    loaded_skills = load_skills_from_folder(skills_folder_input) if skills_folder_input else {}
    if loaded_skills:
        console.print(f"[bold green][Skills Loaded][/bold green] Successfully loaded {len(loaded_skills)} markdown skill profile(s).")
    else:
        console.print("[yellow][Skills Mode][/yellow] No skills folder specified or found. Running in plain agent mode.")

    has_skills = len(loaded_skills) > 0
    if has_skills and is_rag_active:
        mode_label = "[cyan]Skilled Private Agent[/cyan]"
    elif has_skills:
        mode_label = "[cyan]Skilled Agent[/cyan]"
    elif is_rag_active:
        mode_label = "[cyan]Private Agent[/cyan]"
    else:
        mode_label = "[cyan]Agent[/cyan]"

    console.print(f"[bold green][Runtime Mode][/bold green] Active Mode Indicator: {mode_label}")

    workspace_input = WORKSPACE_ROOT_DEFAULT if WORKSPACE_ROOT_DEFAULT != "." else console.input("[cyan]Enter workspace root path for security sandbox (Press Enter for current dir): [/cyan]").strip()
    if workspace_input:
        SandboxManager.set_root(workspace_input)
    console.print(f"[bold green][Security][/bold green] Workspace root locked to: {SandboxManager.root_dir}")

    available_models = fetch_local_chat_models()
    if not available_models:
        console.print("[bold red][Error] No active chat models found via Ollama.[/bold red]")
        sys.exit(1)

    session_id = f"session_{int(time.time())}"

    while True:
        console.print("\n[bold underline]Available Local Chat Models:[/bold underline]")
        for idx, model_name in enumerate(available_models, 1):
            console.print(f"  [cyan]{idx}.[/cyan] {model_name}")

        while True:
            choice_input = console.input(f"[cyan]Select model choice [1-{len(available_models)}] (Default: 1): [/cyan]").strip()
            if not choice_input:
                selected_model = available_models[0]
                break
            try:
                choice_idx = int(choice_input) - 1
                if 0 <= choice_idx < len(available_models):
                    selected_model = available_models[choice_idx]
                    break
            except ValueError:
                pass
            console.print("[red][Error] Invalid selection.[/red]")

        is_reasoning_model = "deepseek-r1" in selected_model.lower()
        console.print(f"[green][Info] Initializing chat model client for: {selected_model}[/green]")

        llm = ChatOllama(model=selected_model, temperature=MODEL_TEMPERATURE, base_url="http://localhost:11434")
        tools_list = list(AVAILABLE_TOOLS.values())
        
        try:
            llm_with_tools = llm.bind_tools(tools_list)
        except Exception as e:
            error_str = str(e).lower()
            if "does not support tools" in error_str or "status code: 400" in error_str or "support tool" in error_str or "primary fails" in error_str or "fails" in error_str:
                console.print(f"[yellow][Model Fallback] Model '{selected_model}' failed tool binding. Attempting automatic fallback...[/yellow]")
                fallback_candidates = [m for m in available_models if "qwen" in m.lower() or "llama" in m.lower()]
                fallback_model = fallback_candidates[0] if fallback_candidates else available_models[0]
                console.print(f"[green][Model Fallback] Switching dynamically to tool-compatible model: {fallback_model}[/green]")
                llm = ChatOllama(model=fallback_model, temperature=MODEL_TEMPERATURE, base_url="http://localhost:11434")
                try:
                    llm_with_tools = llm.bind_tools(tools_list)
                    selected_model = fallback_model
                except Exception as secondary_e:
                    raise secondary_e
            else:
                raise e

        chat_history = memory.load_history(session_id)
        console.print(f"\n[bold green]--- {mode_label} Ready! Type 'exit', 'quit', or 'switch' ---[/bold green]")

        model_selection_failed = False
        try:
            while True:
                user_input = console.input("\n[bold blue]User:[/bold blue] ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    console.print("\n[cyan][Info] Summarizing session and shutting down...[/cyan]")
                    try:
                        summary_prompt = "Summarize the key technical takeaways, code solutions, and user preferences from this session in 2 sentences."
                        summary_res = (await llm.ainvoke([HumanMessage(content=summary_prompt)])).content
                        memory.save_summary(session_id, summary_res)
                        if vectorstore:
                            vectorstore.add_texts([summary_res])
                    except Exception:
                        pass
                    return
                if user_input.lower() == "switch":
                    console.print("[cyan][Info] Returning to model selection...[/cyan]")
                    model_selection_failed = True
                    break
                if not user_input:
                    continue

                active_skill = match_skill_by_relevancy(user_input, loaded_skills) if has_skills else None
                MAX_TOOL_ITERATIONS = active_skill.max_iterations if active_skill else 15

                if active_skill:
                    console.print(f"[magenta][Skill Active][/magenta] {active_skill.name} (Max Tool Budget: {MAX_TOOL_ITERATIONS})")

                context_stats = estimate_context_window(chat_history, user_input)
                console.print(f"[dim][Context State] ~{context_stats['tokens']} / {context_stats['max']} tokens used ({context_stats['percent']}%)[/dim]")

                status_message = "Model is thinking (Chain-of-Thought active)..." if is_reasoning_model else "Generating response..."

                context_text = ""
                past_summaries = memory.get_all_episodic_summaries()
                episodic_context = "\n".join(past_summaries) if past_summaries else ""

                if vectorstore:
                    try:
                        relevant_docs = vectorstore.similarity_search(user_input, k=2)
                        if relevant_docs:
                            context_text = "\n".join([d.page_content for d in relevant_docs])
                            console.print(f"[cyan][RAG State][/cyan] Retrieved {len(relevant_docs)} document/episodic chunks.")
                    except Exception:
                        console.print("[yellow][RAG State] Warning: Vector search failed.[/yellow]")

                enriched_input = f"[Past Episodic Memory & Knowledge Context]:\n{episodic_context}\n{context_text}\n\n[User Query]: {user_input}"

                system_prompts = ["""Role and Identity
You are private_agent, a secure, local-first personal AI collaborator and system assistant. Your primary function is to interpret user queries accurately, leverage your available tooling capabilities, and deliver precise, context-aware answers.

Core Operating Principles
    Truthfulness and Transparency: If you do not know the answer or lack sufficient information in your context or tool outputs, explicitly state: "I don't know." Never fabricate, guess, or hallucinate information.
    Strict Citation Mandate: Any information derived from retrieved sources, memory databases, or tool executions MUST be cited immediately using the format ``. Do not include unverified or irrelevant assertions without supporting data references.
    Tool Utilization: Actively leverage available tools (such as web search, file readers, local SQLite chat history, and document retrievers) whenever a query requires external verification, file access, or historical context.
    Conciseness and Precision: Avoid conversational filler, meta-announcements, or generic preamble. Jump straight into structured, direct answers using bullet points, tables, or concise paragraphs.

Tool Execution Guidelines
    Inspect user queries to determine if tools are required.
    Execute tools with validated arguments. If a tool execution results in an error or traceback, analyze the failure reason, incorporate system reflection prompts, and self-correct your approach before returning a final response.
    Respect privacy boundaries: treat all local user files, database records, and personal context with strict confidentiality.

Response Structure
    Direct Answers First: Lead with the core answer or solution in the opening sentences.
    Scaffolding: Use bullet points and bold section categories to organize multi-part technical details, code snippets, or data summaries.
    Fallback Handling: If a requested action or tool invocation fails completely, inform the user plainly of the limitation without breaking character or outputting malformed data.
                """]
                if active_skill:
                    system_prompts.append(SystemMessage(content=active_skill.system_prompt))

                messages = system_prompts + chat_history + [HumanMessage(content=enriched_input)]

                start_time = time.time()
                try:
                    with console.status(f"[bold cyan]{status_message}[/bold cyan]"):
                        response = await llm_with_tools.ainvoke(messages)
                except Exception as e:
                    error_str = str(e).lower()
                    if "does not support tools" in error_str or "status code: 400" in error_str or "support tool" in error_str:
                        console.print(f"[yellow][Model Fallback] Model '{selected_model}' lacks native tool support. Attempting automatic fallback...[/yellow]")
                        fallback_candidates = [m for m in available_models if "qwen" in m.lower() or "llama" in m.lower()]
                        fallback_model = fallback_candidates[0] if fallback_candidates else available_models[0]
                        console.print(f"[green][Model Fallback] Switching dynamically to tool-compatible model: {fallback_model}[/green]")
                        llm = ChatOllama(model=fallback_model, temperature=MODEL_TEMPERATURE, base_url="http://localhost:11434")
                        llm_with_tools = llm.bind_tools(tools_list)
                        selected_model = fallback_model
                        response = await llm_with_tools.ainvoke(messages)
                    else:
                        raise e

                iteration = 0
                while getattr(response, "tool_calls", None) and iteration < MAX_TOOL_ITERATIONS:
                    iteration += 1
                    console.print(f"[cyan][Agent Loop] Executing tool step {iteration} of {MAX_TOOL_ITERATIONS} concurrently...[/cyan]")
                    messages.append(response)

                    tool_tasks = [execute_tool_call(tc) for tc in response.tool_calls]
                    tool_messages = await asyncio.gather(*tool_tasks)
                    messages.extend(tool_messages)

                    try:
                        with console.status("[bold cyan]Processing tool outputs...[/bold cyan]"):
                            response = await llm_with_tools.ainvoke(messages)
                    except Exception as e:
                        if "does not support tools" in str(e).lower() or "status code: 400" in str(e).lower():
                            raise RuntimeError(f"Model '{selected_model}' does not support tool calling.")
                        raise e

                elapsed_time = time.time() - start_time

                console.print(f"\n[bold green]AI ({selected_model}):[/bold green]")
                full_output_content = ""
                
                async for chunk in llm_with_tools.astream(messages):
                    if chunk.content:
                        sys.stdout.write(chunk.content)
                        sys.stdout.flush()
                        full_output_content += chunk.content
                print()
                console.print(f"[dim green][Latency] Turn completed in {elapsed_time:.2f}s.[/dim green]")

                memory.save_message(session_id, "human", user_input)
                memory.save_message(session_id, "ai", full_output_content)

                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=full_output_content))

        except RuntimeError as rte:
            console.print(f"\n[red][Exception] {str(rte)}[/red]")
            console.print(f"[red][Action Required] Please select a tool-compatible model.[/red]\n")
            continue
        except Exception as ex:
            console.print(f"\n[red][Exception] Unexpected runtime error: {str(ex)}[/red]\n")
            continue

        if model_selection_failed:
            continue
        break

def run_agent_cli():
    asyncio.run(run_agent_cli_async())
