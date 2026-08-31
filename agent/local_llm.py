"""
Local chat backend for support_agent.py (LLM_PROVIDER=local). Runs Qwen2.5-7B
as a Q4_K_M GGUF through llama-cpp-python on Metal, no API key or rate limit.

Skips langchain_huggingface's ChatHuggingFace since it drops bound tools from
the prompt for local models and its _agenerate isn't implemented. Instead we
call tokenizer.apply_chat_template(..., tools=...) ourselves and parse the
model's <tool_call>{...}</tool_call> output into AIMessage.tool_calls.
"""
import asyncio
import atexit
import json
import re
import uuid
from typing import Any, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

TOKENIZER_ID = "Qwen/Qwen2.5-7B-Instruct"
GGUF_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"

# two shards on the HF repo -- llama.cpp picks up the second one automatically
# as long as both files sit in the same directory
GGUF_FILES = [
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
]

cached_tokenizer = cached_llm = None


def load_model(n_ctx: int):
    global cached_tokenizer, cached_llm
    if cached_llm is None:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        from transformers import AutoTokenizer

        shard_paths = [hf_hub_download(repo_id=GGUF_REPO, filename=f) for f in GGUF_FILES]
        cached_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        cached_llm = Llama(model_path=shard_paths[0], n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
        # llama-cpp-python hits a Metal residency-set assertion on normal exit
        # without this
        atexit.register(cached_llm.close)
    return cached_tokenizer, cached_llm


ROLE_BY_MESSAGE_TYPE = {"system": "system", "human": "user", "ai": "assistant"}


def to_chatml(messages: Sequence[BaseMessage]) -> List[dict]:
    out = []
    for m in messages:
        if m.type == "tool":
            out.append({"role": "tool", "name": m.name, "content": str(m.content)})
        elif m.type == "ai" and getattr(m, "tool_calls", None):
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": tc["name"], "arguments": tc["args"]}}
                        for tc in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": ROLE_BY_MESSAGE_TYPE[m.type], "content": m.content})
    return out


class LocalQwenChat(BaseChatModel):
    """Local quantized Qwen2.5-7B-Instruct via llama-cpp-python."""

    n_ctx: int = 8192
    temperature: float = 0.0
    max_new_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "local-qwen-gguf-chat"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        tokenizer, llm = load_model(self.n_ctx)
        tools = kwargs.get("tools")
        text = tokenizer.apply_chat_template(
            to_chatml(messages), tools=tools or None, add_generation_prompt=True, tokenize=False
        )
        result = llm(text, max_tokens=self.max_new_tokens, temperature=self.temperature, stop=["<|im_end|>"])
        raw = result["choices"][0]["text"]

        tool_calls = []
        for match in TOOL_CALL_RE.finditer(raw):
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            tool_calls.append(
                {
                    "name": data.get("name"),
                    "args": data.get("arguments", {}),
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "tool_call",
                }
            )
        content = TOOL_CALL_RE.sub("", raw).strip() if tool_calls else raw

        usage = result.get("usage") or {}
        message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # no async path in llama-cpp-python, so just offload to a thread
        return await asyncio.to_thread(self._generate, messages, stop, None, **kwargs)

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> Runnable:
        """JSON mode via prompting instead of tool-calling -- simpler and more
        reliable at this model size."""
        parser = PydanticOutputParser(pydantic_object=schema)
        instructions = parser.get_format_instructions()

        def augment(input_: Any) -> List[BaseMessage]:
            if isinstance(input_, str):
                return [HumanMessage(content=f"{input_}\n\n{instructions}")]
            messages = list(input_)
            messages.append(HumanMessage(content=instructions))
            return messages

        return RunnableLambda(augment) | self | StrOutputParser() | parser
