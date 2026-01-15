#!/usr/bin/env python3
"""
FastAPI后端服务 - 为ReAct Agent提供Web接口（支持流式输出）
"""

import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import asyncio

from agent import ReactAgent, logger

# 全局Agent实例
agent: Optional[ReactAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global agent
    try:
        agent = ReactAgent()
    except ValueError as e:
        logger.error(f"Agent初始化失败: {e}")
    yield
    logger.info("应用关闭")


app = FastAPI(
    title="ReAct Agent API",
    description="基于LangChain的ReAct模式智能Agent（支持流式输出）",
    version="2.0.0",
    lifespan=lifespan
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    success: bool


class ToolInfo(BaseModel):
    name: str
    description: str


@app.get("/")
async def root():
    """返回聊天页面"""
    return FileResponse("static/index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理聊天请求（非流式）"""
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


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式处理聊天请求（SSE）"""
    global agent
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent未初始化，请检查环境变量")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    
    async def generate():
        try:
            for chunk in agent.chat_stream(request.message):
                if chunk:
                    # SSE格式
                    yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"流式输出错误: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
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
