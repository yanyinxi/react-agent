#!/usr/bin/env python3
"""
FastAPI后端服务 - 为ReAct Agent提供Web接口
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from agent import ReactAgent

app = FastAPI(
    title="ReAct Agent API",
    description="基于LangChain的ReAct模式智能Agent",
    version="1.0.0"
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局Agent实例
agent: Optional[ReactAgent] = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    success: bool


class ToolInfo(BaseModel):
    name: str
    description: str


@app.on_event("startup")
async def startup_event():
    """启动时初始化Agent"""
    global agent
    try:
        agent = ReactAgent()
        print("ReAct Agent 初始化成功")
        print("可用工具:", [t["name"] for t in agent.get_tools_info()])
    except ValueError as e:
        print(f"Agent初始化失败: {e}")
        print("请确保设置了环境变量 OPENAI_BASE_URL 和 OPENAI_API_KEY")


@app.get("/")
async def root():
    """返回聊天页面"""
    return FileResponse("static/index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理聊天请求"""
    global agent
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent未初始化，请检查环境变量")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    
    result = agent.chat(request.message)
    return ChatResponse(
        response=result["output"],
        success=result["success"]
    )


@app.post("/api/reset")
async def reset():
    """重置对话历史"""
    global agent
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent未初始化")
    
    agent.reset()
    return {"status": "ok", "message": "对话已重置"}


@app.get("/api/tools", response_model=List[ToolInfo])
async def get_tools():
    """获取可用工具列表"""
    global agent
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent未初始化")
    
    return agent.get_tools_info()


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "agent_ready": agent is not None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
