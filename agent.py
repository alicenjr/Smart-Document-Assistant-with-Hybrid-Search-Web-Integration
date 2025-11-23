import inspect
import os
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
import random
from datetime import datetime
import requests
from generation import generate_rag_response

# --- Math Tools ---
class MultiplyInput(BaseModel):
    a: int = Field(..., description="The first number")
    b: int = Field(..., description="The second number")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput
    def _run(self, a: int, b: int) -> int:
        return a * b


class AddRandom(BaseModel):
    a: int = Field(..., description="A number to add a random value to")

class AddRandomTool(BaseTool):
    name: str = "random_adder"
    description: str = "Add a random number (1–100) to the input"
    args_schema: Type[BaseModel] = AddRandom
    def _run(self, a: int) -> int:
        return a + random.randint(1, 100)


# --- Date & Time Tool ---
class CurrentDateTimeTool(BaseTool):
    name: str = "get_current_datetime"
    description: str = "Returns the current date and time as a formatted string."
    def _run(self) -> str:
        now = datetime.now()
        return now.strftime("%A, %d %B %Y, %H:%M:%S")


# --- Serper Web Search Tool ---
class SerperSearchInput(BaseModel):
    query: str = Field(..., description="Search query for up-to-date web information.")


class SerperSearchTool(BaseTool):
    name: str = "serper_web_search"
    description: str = (
        "Search the public web via Serper when information is not available in the knowledge base."
    )
    args_schema: Type[BaseModel] = SerperSearchInput

    def _run(self, query: str) -> str:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            api_key = "ffa0120b601f768440f6f2bb82289fee7d239d9b"  # TODO: replace with secure config before deployment
        if not api_key:
            return (
                "Serper API key missing. Set SERPER_API_KEY environment variable to enable web search."
            )

        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query},
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            return f"Serper web search failed: {exc}"

        organic = []
        if isinstance(payload, dict):
            organic = payload.get("organic", []) or []

        if not organic:
            return "Serper search returned no organic results."

        lines = []
        for item in organic[:5]:
            title = item.get("title", "Untitled result")
            snippet = item.get("snippet") or item.get("link", "")
            link = item.get("link", "")
            lines.append(f"- {title}\n  {snippet}\n  Source: {link}")

        return "Serper web search results:\n" + "\n".join(lines)


# --- RAG Retrieval Tool ---
class RagQueryInput(BaseModel):
    query: str = Field(..., description="User question to retrieve and answer from the knowledge base.")
    search_type: str = Field(
        default="hybrid",
        description="Search mode to use: keyword, semantic, or hybrid."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of chunks to retrieve."
    )
    model_type: str = Field(
        default="gemini",
        description="Model backend for generation. Default is gemini."
    )
    stream: bool = Field(
        default=True,
        description="Whether to stream the response to stdout while retrieving."
    )


class RagQueryTool(BaseTool):
    name: str = "rag_query"
    description: str = (
        "Use this to answer questions grounded in the ingested knowledge base. "
        "Provide the user's question; optionally adjust search_type (keyword, semantic, hybrid) or top_k."
    )
    args_schema: Type[BaseModel] = RagQueryInput

    def _run(
        self,
        query: str,
        search_type: str = "hybrid",
        top_k: int = 5,
        model_type: str = "gemini",
        stream: bool = True,
    ) -> str:
        try:
            if stream:
                generator = generate_rag_response(
                    query=query,
                    search_type=search_type,
                    top_k=top_k,
                    model_type=model_type,
                    stream=True,
                )
                chunks: list[str] = []
                streamed_any = False
                for chunk in generator:
                    if not chunk:
                        continue
                    print(chunk, end="", flush=True)
                    chunks.append(str(chunk))
                    streamed_any = True
                if streamed_any:
                    print()
                raw_response = "".join(chunks)
            else:
                raw_response = generate_rag_response(
                    query=query,
                    search_type=search_type,
                    top_k=top_k,
                    model_type=model_type,
                    stream=False,
                )
            response = raw_response

            while True:
                if inspect.isgenerator(response):
                    iterator = response
                elif hasattr(response, "__iter__") and not isinstance(
                    response, (str, bytes, dict)
                ):
                    iterator = iter(response)
                else:
                    break

                try:
                    first_chunk = next(iterator)
                except StopIteration as stop:
                    response = stop.value
                    continue
                else:
                    chunks = []
                    if first_chunk:
                        chunks.append(str(first_chunk))
                    for chunk in iterator:
                        if chunk:
                            chunks.append(str(chunk))
                    response = "".join(chunks)
                    break
                finally:
                    try:
                        close_method = getattr(iterator, "close", None)
                        if callable(close_method):
                            close_method()
                    except Exception:
                        pass

            if isinstance(response, (list, tuple)):
                response = "".join(str(part) for part in response if part)
            elif isinstance(response, dict):
                response = str(response)

            if not response:
                return "No relevant information found in the knowledge base."
            return str(response)
        except Exception as exc:
            return f"RAG query failed: {exc}"


# --- Register all tools ---
tools = [
    RagQueryTool(),
    AddRandomTool(),
    MultiplyTool(),
    CurrentDateTimeTool(),
    SerperSearchTool(),
]

# --- LLM (Gemini) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = "AIzaSyCmOMbb1ifcOzxrvRx1NAdz_o248fjLIic"  # TODO: replace with secure configuration for production

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_output_tokens=2048,
)

SYSTEM_PROMPT = """You are a precise and polite AI assistant that can use tools step-by-step.

TOOLS AVAILABLE:
1. rag_query(query, search_type?, top_k?, model_type?, stream?): retrieves context from the knowledge base and answers using the RAG pipeline. Streaming is enabled by default; disable it only if the user requests a single-shot answer.
2. random_adder(a): adds a random number (1–100) to the input.
3. multiply(a, b): multiplies two numbers.
4. get_current_datetime(): returns the current local date and time.
5. serper_web_search(query): searches the public web via Serper when the knowledge base lacks the information.

RULES:
- For any question about factual content, stored documents, or knowledge base topics, you MUST call rag_query exactly once before giving a final answer.
- If rag_query returns no useful information, say so explicitly.
- Only call serper_web_search when the user explicitly asks for external/current information or when rag_query yields no relevant data, then summarize the retrieved web findings.
- Use math tools for calculations.
- Use get_current_datetime for any date/time questions.
- Always call one tool at a time (no nested tool calls).
- Final answers should be clearly written, human-like, and concise.
"""

# --- Create the agent graph ---
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    debug=True,
)

# --- Example runs ---

if __name__ == "__main__":
    demo_result = agent.invoke({
        "messages": [
            {"role": "user", "content": "tell me about mewing and its benefits"}
        ]
    })

    messages = demo_result.get("messages", [])
    final_message = messages[-1] if messages else None

    # Determine which tools were used during the run
    tool_names: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            name = getattr(message, "name", None)
            if name and name not in tool_names:
                tool_names.append(name)

    # Identify model used (falling back to configured model)
    model_used = None
    if final_message and hasattr(final_message, "response_metadata"):
        model_used = final_message.response_metadata.get("model_name")
    if not model_used:
        model_used = getattr(llm, "model", "unknown-model")

    # Fallback: if no tools fired, run rag_query manually for the user's first message
    if not tool_names:
        user_query = None
        for message in messages:
            if isinstance(message, HumanMessage):
                user_query = message.content
                if isinstance(user_query, list):
                    user_query = "\n".join(
                        part.get("text", "")
                        if isinstance(part, dict)
                        else str(part)
                        for part in user_query
                    )
                user_query = str(user_query)
                break
        if user_query:
            print("Agent skipped tool calls; invoking rag_query manually...\n")
            manual_tool = RagQueryTool()
            manual_output = manual_tool._run(query=user_query)
            tool_names.append(manual_tool.name)
            final_message = None
        else:
            manual_output = None
    else:
        manual_output = None

    # Extract final content
    if manual_output is not None:
        content = manual_output
    elif final_message:
        content = final_message.content
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    else:
        content = "No response returned."

    print("\n--- Run Summary ---")
    print(f"Model used: {model_used}")
    print(f"Tools used: {', '.join(tool_names) if tool_names else 'None'}")
    print("Content:")
    print(content)
