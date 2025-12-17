import os,logging,warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False")
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
os.environ.update({
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "CUDA_VISIBLE_DEVICES": "1"
})
logging.getLogger("tensorflow").setLevel(logging.FATAL)

import torch
import asyncio
import importlib
from typing import Optional, List, Dict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModel
)
from llama_index.core import (
    Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("未安装FastAPI依赖，仅能运行本地测试模式")

# ===================== 全局变量 & 自定义参数配置 =====================
agent = None
custom_func = []
custom_module_name = ""

EMBEDDING_MODEL_PATH = ""
LLM_MODEL_PATH = ""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,  # 改为 load_in_8bit=True 可切换8bit量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
    bnb_4bit_quant_type="nf4",  # 4bit量化类型（nf4适合LLM）
    bnb_4bit_use_double_quant=True,
    bnb_4bit_bnb_quant_storage=torch.uint8
)

TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MAX_NEW_TOKENS = 1024
CONTEXT_WINDOW = 8192
MEMORY_CACHE_LENGTH = 4090
RAG_DOCUMENT_DIR = r"./knowledge"

# 提示词模板
QA_PROMPT_TEMPLATE_STR = (
    "You are a helpful assistant. Please answer the user's question based on the provided context.\n\n"
    "Context: {context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)

# 系统提示词
system_prompt = """
你是一位专业的金融投资顾问，严格遵守以下规则：
1. 优先使用financial_principle_knowledge工具获取内部金融文档的权威信息；
2. 根据用户的风险偏好给出个性化、负责任的投资建议；
3. 多轮对话中需记住用户之前的提问和你的回答，保持上下文连贯。
"""

# ===================== 核心Agent初始化函数 =====================
def init_agent():
    global agent
    if custom_module_name:
        custom_module = importlib.import_module(custom_module_name)
    else:
        print("未获取到ML分析模块！")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_PATH,
        device=DEVICE,
        normalize=True,
        max_length=512,
        model_kwargs={
            "torch_dtype": TORCH_DTYPE,
            "device_map": "auto",
            "load_in_4bit": False,
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_PATH,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=QUANTIZATION_CONFIG,
    )

    Settings.llm = HuggingFaceLLM(
        model=llm_model,
        tokenizer=tokenizer,
        context_window=CONTEXT_WINDOW,
        max_new_tokens=MAX_NEW_TOKENS,
        system_prompt=system_prompt,
        generate_kwargs={
            "do_sample": False,
            "num_beams": 1,
        },
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=MEMORY_CACHE_LENGTH)

    documents = SimpleDirectoryReader(RAG_DOCUMENT_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    qa_prompt_template = PromptTemplate(QA_PROMPT_TEMPLATE_STR)
    query_engine = index.as_query_engine(text_qa_template=qa_prompt_template)
    rag_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="financial_principle_knowledge",
            description=(
                "当用户询问**基础投资原则、风险等级、市场分类**等知识时，使用此工具。 "
                "它提供来自内部金融文档的背景信息和规则。"
            ),
        ),
    )
    ml_analysis_tools=[]
    for funcs in custom_func:
        ml_analysis_tools.append(FunctionTool.from_defaults(fn=getattr(custom_module, funcs)))

    #ReAct Agent
    agent = ReActAgent(
        tools=[rag_tool] + ml_analysis_tools,
        llm=Settings.llm,
        verbose=True,
        memory=memory,
        system_prompt=system_prompt,
        max_iterations=20,
        allow_parallel_tool_calls=False,
        handle_parsing_errors=True,
        return_first_response=True,
    )

# ===================== FastAPI =====================
def start_agent_api_server():
    app = FastAPI(title="金融Agent API服务")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
    )
    class ChatRequest(BaseModel):
        prompt: str
        history: Optional[List[Dict[str, str]]] = []
    @app.on_event("startup")
    async def startup_event():
        init_agent()

    @app.post("/chat")
    async def chat(request: ChatRequest):
        try:
            global agent
            history = request.history.copy()
            history.append({"role": "user", "content": request.prompt})
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in history)

            handler = agent.run(prompt)
            if asyncio.iscoroutine(handler):
                handler = await handler
            buffer = ""
            async for ev in handler.stream_events():
                text = None
                if hasattr(ev, "response") and ev.response:
                    text = ev.response
                elif hasattr(ev, "message"):
                    msg = ev.message
                    text = msg.content if hasattr(msg, "content") else str(msg)
                elif ev.__class__.__name__ in ["AgentOutput", "StopEvent"]:
                    try:
                        text = ev.response if hasattr(ev, "response") else ev.result()
                    except:
                        pass
                if text:
                    buffer = str(text)

            new_history = history + [{"role": "assistant", "content": buffer}]
            return {
                "code": 200,
                "msg": "success",
                "data": {
                    "response": buffer,
                    "history": new_history
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent调用失败：{str(e)}")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        workers=1,
        reload=False
    )

if __name__ == "__main__":
    start_agent_api_server()

