#!/usr/bin/env python3
"""
FastAPI后端服务 - 为ReAct Agent提供Web接口（支持流式输出和会话管理）
"""

import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Cookie, Response
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import uvicorn

from agent import ReactAgent, logger, AgentConfig


# ============ 配置常量 ============

MAX_MESSAGE_LENGTH = 10000  # 最大消息长度
SESSION_TIMEOUT = 3600  # 会话超时时间（秒）
MAX_SESSIONS = 100  # 最大会话数


# ============ 会话管理 ============

class SessionManager:
    """会话管理器，为每个用户维护独立的 Agent 实例"""

    def __init__(self, max_sessions: int = MAX_SESSIONS, timeout: int = SESSION_TIMEOUT):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._timeout = timeout

    def get_or_create_agent(self, session_id: str) -> ReactAgent:
        """获取或创建会话对应的 Agent"""
        with self._lock:
            now = time.time()

            # 清理过期会话
            self._cleanup_expired(now)

            if session_id in self._sessions:
                # 更新访问时间
                self._sessions[session_id]["last_access"] = now
                return self._sessions[session_id]["agent"]

            # 如果会话数已满，删除最旧的会话
            if len(self._sessions) >= self._max_sessions:
                self._remove_oldest()

            # 创建新会话
            agent = ReactAgent()
            self._sessions[session_id] = {
                "agent": agent,
                "created_at": now,
                "last_access": now
            }
            logger.info(f"创建新会话: {session_id[:8]}...")
            return agent

    def reset_session(self, session_id: str) -> bool:
        """重置指定会话的对话历史"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["agent"].reset()
                self._sessions[session_id]["last_access"] = time.time()
                return True
            return False

    def get_agent(self, session_id: str) -> Optional[ReactAgent]:
        """获取指定会话的 Agent（不创建新会话）"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["last_access"] = time.time()
                return self._sessions[session_id]["agent"]
            return None

    def _cleanup_expired(self, now: float) -> None:
        """清理过期会话"""
        expired = [
            sid for sid, data in self._sessions.items()
            if now - data["last_access"] > self._timeout
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.info(f"会话过期已删除: {sid[:8]}...")

    def _remove_oldest(self) -> None:
        """删除最旧的会话"""
        if not self._sessions:
            return
        oldest_sid = min(
            self._sessions.keys(),
            key=lambda sid: self._sessions[sid]["last_access"]
        )
        del self._sessions[oldest_sid]
        logger.info(f"会话数已满，删除最旧会话: {oldest_sid[:8]}...")

    @property
    def session_count(self) -> int:
        """当前会话数"""
        with self._lock:
            return len(self._sessions)


# 全局会话管理器
session_manager: Optional[SessionManager] = None


# ============ 应用生命周期 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global session_manager
    try:
        session_manager = SessionManager()
        logger.info("会话管理器初始化完成")
    except Exception as e:
        logger.error(f"会话管理器初始化失败: {e}")
    yield
    logger.info("应用关闭")


# ============ FastAPI 应用 ============

app = FastAPI(
    title="ReAct Agent API",
    description="基于LangChain的ReAct模式智能Agent（支持流式输出和会话隔离）",
    version="3.0.0",
    lifespan=lifespan
)

# CORS 配置（限制来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",  # 开发环境前端
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ============ 请求/响应模型 ============

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ChatResponse(BaseModel):
    response: str
    success: bool


class ToolInfo(BaseModel):
    name: str
    description: str


class HealthResponse(BaseModel):
    status: str
    session_count: int


class ResetResponse(BaseModel):
    status: str
    message: str


# ============ 辅助函数 ============

def get_session_id(session_id: Optional[str]) -> str:
    """获取或生成会话 ID"""
    return session_id if session_id else str(uuid4())


def validate_message(message: str) -> None:
    """验证消息"""
    if not message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail=f"消息过长，最大 {MAX_MESSAGE_LENGTH} 字符")


# ============ API 端点 ============

@app.get("/")
async def root() -> FileResponse:
    """返回聊天页面"""
    return FileResponse("static/index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    response: Response,
    session_id: Optional[str] = Cookie(default=None)
) -> ChatResponse:
    """处理聊天请求（非流式）"""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="服务未初始化")

    validate_message(request.message)

    # 获取或创建会话
    sid = get_session_id(session_id)
    if not session_id:
        response.set_cookie(key="session_id", value=sid, httponly=True, samesite="lax")

    try:
        agent = session_manager.get_or_create_agent(sid)
        result = agent.chat(request.message)
        return ChatResponse(
            response=result["output"],
            success=result["success"]
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    response: Response,
    session_id: Optional[str] = Cookie(default=None)
) -> StreamingResponse:
    """流式处理聊天请求（SSE）"""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="服务未初始化")

    validate_message(request.message)

    # 获取或创建会话
    sid = get_session_id(session_id)

    try:
        agent = session_manager.get_or_create_agent(sid)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    async def generate():
        try:
            for chunk in agent.chat_stream(request.message):
                if chunk:
                    yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except GeneratorExit:
            # 客户端断开连接
            logger.info(f"客户端断开连接: {sid[:8]}...")
        except Exception as e:
            logger.error(f"流式输出错误: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    streaming_response = StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

    # 设置 cookie
    if not session_id:
        streaming_response.set_cookie(key="session_id", value=sid, httponly=True, samesite="lax")

    return streaming_response


@app.post("/api/reset", response_model=ResetResponse)
async def reset(session_id: Optional[str] = Cookie(default=None)) -> ResetResponse:
    """重置对话历史"""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="服务未初始化")

    if not session_id:
        raise HTTPException(status_code=400, detail="无有效会话")

    if session_manager.reset_session(session_id):
        return ResetResponse(status="ok", message="对话已重置")
    else:
        raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/api/tools", response_model=List[ToolInfo])
async def get_tools(session_id: Optional[str] = Cookie(default=None)) -> List[ToolInfo]:
    """获取可用工具列表"""
    if session_manager is None:
        raise HTTPException(status_code=500, detail="服务未初始化")

    # 获取任意一个 agent 的工具信息（所有 agent 工具相同）
    sid = get_session_id(session_id)
    try:
        agent = session_manager.get_or_create_agent(sid)
        return [ToolInfo(**info) for info in agent.get_tools_info()]
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """健康检查"""
    return HealthResponse(
        status="healthy",
        session_count=session_manager.session_count if session_manager else 0
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
