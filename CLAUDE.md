# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 项目概述

这是一个基于 LangChain 和 FastAPI 构建的 ReAct（推理+行动）Agent。提供命令行界面和 Web API 两种方式，与可使用工具解决问题的 AI 助手进行交互。

## 常用命令

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动 Web 服务
```bash
python app.py
```
服务运行在 `http://0.0.0.0:8000`。

### 命令行模式
```bash
python agent.py
```
CLI 命令：`quit`（退出）、`reset`（重置对话）、`tools`（查看工具列表）。

## 必需的环境变量

- `OPENAI_BASE_URL` - OpenAI 兼容 API 的基础 URL（需完整 URL 格式）
- `OPENAI_API_KEY` - API 认证密钥（长度需 >= 10）

## 架构

### 核心组件

**agent.py** - 包含核心类：
- `AgentConfig` - Agent 配置（模型、温度、超时、历史长度等）
- `SafeMathEvaluator` - 安全的数学表达式计算器（使用 AST 解析，防止代码注入）
- `ReactAgent` - 主 Agent 类：
  - 封装 LangChain 的 `create_react_agent`（来自 `langgraph.prebuilt`）
  - 使用线程锁保护对话历史（`threading.Lock`）
  - 对话历史自动裁剪（默认最多 50 条，防止内存泄漏）
  - 提供同步（`chat()`）和流式（`chat_stream()`）接口
  - 分类异常处理（`RateLimitError`、`APITimeoutError`、`APIError`）

**app.py** - FastAPI 服务端：
- `SessionManager` - 会话管理器（每个用户独立 Agent 实例）
  - 基于 Cookie 的会话跟踪
  - 自动过期清理（默认 1 小时）
  - 最大会话数限制（默认 100）
- API 端点：
  - `POST /api/chat` - 非流式聊天
  - `POST /api/chat/stream` - SSE 流式聊天
  - `POST /api/reset` - 重置对话
  - `GET /api/tools` - 工具列表
  - `GET /api/health` - 健康检查

### 工具（定义在 agent.py）

工具使用 `langchain_core.tools` 的 `@tool` 装饰器定义：
- `calculator` - 安全数学计算（支持 +,-,*,/,**,%，以及 sqrt,sin,cos,tan,log 等函数）
- `get_current_time` - 获取当前日期时间
- `search_web` - 模拟网络搜索（占位实现）
- `text_analyzer` - 文本统计（字符数/词数/行数）

### 添加新工具

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """工具描述，会展示给 LLM"""
    return "result"
```

然后在 `ReactAgent.__init__` 中将工具添加到 `self.tools` 列表。

### 配置调整

通过 `AgentConfig` 类调整参数：
```python
config = AgentConfig(
    model="claude-opus-4-5-20251101",
    temperature=0,
    max_tokens=2000,
    max_history=50,  # 对话历史条数
    timeout=30       # API 超时秒数
)
agent = ReactAgent(config=config)
```
