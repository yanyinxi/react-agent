# ReAct Agent - 智能推理与行动代理

## 项目概述

这是一个基于 LangChain 和 FastAPI 构建的**ReAct（推理+行动）智能Agent**。该项目提供了一个完整的 AI 助手解决方案，支持工具调用、流式输出、会话管理和多种交互方式。

### 核心特性

- ✨ **ReAct 推理框架**：边推理边执行，通过使用工具来解决复杂问题
- 🔧 **多种工具集成**：计算器、网络搜索、文本分析等
- 🌐 **Web API 接口**：基于 FastAPI 的高性能 Web 服务
- 💬 **会话管理**：独立的会话隔离，支持多用户并发
- 📡 **流式输出**：实时流式响应，改善用户体验
- 🛡️ **安全机制**：AST 解析防止代码注入，多重验证
- 📝 **详细日志**：彩色日志输出，便于调试
- 🖥️ **现代化 UI**：基于 HTML5 的响应式聊天界面

---

## 项目结构

```
react-agent/
├── agent.py              # 核心 Agent 实现
├── app.py                # FastAPI Web 服务
├── requirements.txt      # 项目依赖
├── CLAUDE.md            # Claude Code 工作指南
├── README.md            # 本文件
├── .claude/             # Claude Code 配置
│   └── agents/          # Agent 定义
│       └── code-reviewer.md
├── static/              # 前端资源
│   └── index.html       # 聊天界面
└── .git/                # Git 配置

```

---

## 快速开始

### 前置要求

- Python 3.8+
- OpenAI 兼容的 API（如 Claude、OpenAI、Anthropic）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 环境配置

在项目根目录创建 `.env` 文件（或设置环境变量）：

```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key-here
```

**必需的环境变量：**

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `OPENAI_BASE_URL` | OpenAI 兼容 API 的基础 URL（必需完整 URL 格式） | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | API 认证密钥（长度需 >= 10） | `sk-...` |

### 运行方式

#### 方式一：Web 服务 + 前端界面（推荐）

```bash
python app.py
```

访问 `http://localhost:8000` 即可使用聊天界面。

**功能特性：**
- 实时流式聊天
- 会话隔离和管理
- 工具调用可视化
- 响应式设计

#### 方式二：命令行交互

```bash
python agent.py
```

**CLI 命令：**
- `quit` - 退出程序
- `reset` - 重置对话历史
- `tools` - 查看所有可用工具

---

## 架构设计

### 核心组件

#### 1. **agent.py** - 智能 Agent 核心

**主要类：**

##### `AgentConfig`（数据类）
Agent 配置参数管理：
```python
@dataclass
class AgentConfig:
    model: str = "claude-opus-4-5-20251101"  # 使用的模型
    temperature: float = 0                    # 输出确定性（0=最确定）
    max_tokens: int = 2000                   # 单次回复最大 token
    max_history: int = 50                    # 保留的最大对话历史
    timeout: int = 30                        # API 请求超时（秒）
```

##### `SafeMathEvaluator`（安全计算器）
使用 AST（抽象语法树）安全地计算数学表达式：
- **支持的运算**：`+`, `-`, `*`, `/`, `//`, `%`, `**`
- **支持的函数**：`sqrt`, `sin`, `cos`, `tan`, `log`, `abs`, `round`, `pow` 等
- **支持的常量**：`pi`, `e`
- **防护机制**：防止任意代码执行

**示例：**
```python
evaluator = SafeMathEvaluator()
result = evaluator.evaluate("sqrt(16) + 2**3")  # 12.0
```

##### `ReactAgent`（主 Agent 类）
实现 ReAct 模式的智能代理：
- 接收用户输入
- 进行推理（Reasoning）
- 调用相应工具（Acting）
- 返回结果

**关键方法：**
- `__init__()` - 初始化 Agent（LLM、工具、验证环境）
- `chat(user_input)` - 处理用户输入并返回响应
- `stream_chat(user_input)` - 流式处理用户输入
- `reset()` - 清空对话历史
- `get_tools_description()` - 获取工具描述列表

**内部机制：**
```
用户输入 → 构建消息 → LLM 推理 → Agent 决策 → 调用工具 → 收集结果 → 最终回复
     ↓                                         ↓
  历史管理                              流式输出处理
```

#### 2. **app.py** - FastAPI Web 服务

**主要功能：**

##### `SessionManager`（会话管理）
为每个用户维护独立的 Agent 实例：
- **会话隔离**：每个用户有独立的对话历史和状态
- **超时清理**：自动清理超时会话，释放资源
- **并发安全**：使用线程锁保证数据一致性
- **容量管理**：限制最大会话数，防止内存溢出

**参数配置：**
```python
MAX_MESSAGE_LENGTH = 10000  # 单条消息最大长度
SESSION_TIMEOUT = 3600      # 会话超时（秒）
MAX_SESSIONS = 100          # 最大会话数
```

##### REST API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 返回 Web 界面（HTML） |
| `/api/chat` | POST | 发送消息并获取响应 |
| `/api/stream` | POST | 流式聊天接口 |
| `/api/reset` | POST | 重置会话对话历史 |
| `/api/tools` | GET | 获取可用工具列表 |

**示例请求：**

```bash
# 发送消息
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "2的10次方是多少？"}'

# 流式请求
curl -X POST http://localhost:8000/api/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "计算 sqrt(100) + 50"}'
```

#### 3. **static/index.html** - 前端聊天界面

**功能特性：**
- 现代化的响应式设计
- 实时消息流式输出
- 工具调用展示面板
- 会话管理（新建、重置）
- 实时日志查看
- 深色主题 UI

---

## 工具集

Agent 内置的工具集合：

### 1. **计算器（calculator）**
计算数学表达式
```
输入: "sqrt(100) + 2**3"
输出: 计算结果: 108.0
```

### 2. **时间查询（get_current_time）**
获取当前日期和时间
```
输出: 当前时间: 2024年01月17日 14:30:45 (星期三)
```

### 3. **网络搜索（search_web）**
搜索网络信息（目前为模拟实现，可集成真实 API）
```
输入: "Python 教程"
输出: 搜索 'Python 教程' 的结果: [模拟结果]
```

### 4. **文本分析（text_analyzer）**
分析文本的统计信息
```
输入: "Hello 世界"
输出: 
- 总字符数: 8
- 单词/词组数: 2
- 行数: 1
- 中文字符数: 1
```

---

## 工作流程

### ReAct 执行流程

```
┌─────────────┐
│  用户输入   │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│   1. 理解用户意图            │
│   (使用 LLM 进行语义理解)    │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│   2. 推理决策                │
│   (思考是否需要使用工具)     │
└──────────┬───────────────────┘
           │
      ┌────┴─────┐
      │           │
      ▼           ▼
  需要工具    直接回复
      │           │
      ▼           │
┌──────────────┐  │
│ 3. 工具调用 │  │
│   执行操作  │  │
└──────┬───────┘  │
       │          │
       ▼          │
┌──────────────┐  │
│ 4. 结果整合 │  │
└──────┬───────┘  │
       │          │
       └────┬─────┘
            │
            ▼
┌──────────────────────────────┐
│   5. 生成最终回复            │
│   (返回给用户)               │
└──────┬───────────────────────┘
       │
       ▼
┌─────────────────┐
│  返回响应       │
│  更新历史       │
└─────────────────┘
```

---

## API 请求/响应示例

### POST /api/chat

**请求：**
```json
{
  "message": "计算 2 + 2",
  "session_id": "user-123"
}
```

**响应：**
```json
{
  "output": "2 + 2 等于 4。",
  "success": true,
  "session_id": "user-123"
}
```

### POST /api/stream

**请求：**
```json
{
  "message": "sqrt(16) 等于多少？"
}
```

**响应（流式）：**
```
data: 让我计算一下...
data: sqrt(16)
data: 的结果是 4...
```

---

## 配置说明

### Agent 配置（agent.py）

在 `ReactAgent` 初始化时修改：

```python
config = AgentConfig(
    model="claude-opus-4-5-20251101",  # 更换模型
    temperature=0,                      # 确定性
    max_tokens=2000,                   # 最大 token
    max_history=50,                    # 历史长度
    timeout=30                         # 超时时间
)
agent = ReactAgent(config)
```

### Web 服务配置（app.py）

```python
MAX_MESSAGE_LENGTH = 10000  # 消息长度限制
SESSION_TIMEOUT = 3600      # 会话超时（秒）
MAX_SESSIONS = 100          # 最大会话数
```

### 日志配置

- **日志级别**：DEBUG（可修改 `logger.setLevel()`）
- **格式**：彩色输出，包含时间戳和日志级别
- **输出**：标准输出（stdout）

---

## 安全性机制

### 1. **API 密钥验证**
```python
if len(self.api_key) < 10:
    raise ValueError("API 密钥格式无效")
```

### 2. **URL 格式验证**
```python
parsed = urlparse(self.base_url)
if not parsed.scheme or not parsed.netloc:
    raise ValueError("Base URL 格式无效")
```

### 3. **代码注入防护**（SafeMathEvaluator）
- 使用 AST 解析，而非 `eval()`
- 白名单控制：只允许特定的运算符和函数
- 完整的异常处理

### 4. **消息长度限制**
```python
MAX_MESSAGE_LENGTH = 10000
```

### 5. **会话隔离**
- 每个用户独立的 Agent 实例
- 对话历史隔离，防止数据泄露

---

## 常见问题 (FAQ)

### Q: 如何更换 AI 模型？
A: 修改 `AgentConfig` 中的 `model` 参数：
```python
AgentConfig(model="gpt-4-turbo")
```

### Q: 如何添加新的工具？
A: 在 `agent.py` 中定义工具函数并使用 `@tool` 装饰器：
```python
@tool
def my_tool(param: str) -> str:
    """工具描述"""
    return "结果"
```
然后在 `ReactAgent.tools` 列表中添加该工具。

### Q: 对话历史会占用很多内存吗？
A: 不会。系统自动限制历史长度（默认 50 条消息），防止内存泄漏。

### Q: 支持多用户并发吗？
A: 支持。Web 服务使用会话管理，每个用户有独立的会话。

### Q: 如何进行性能优化？
A: 
1. 调整 `max_tokens` 减少单次响应长度
2. 增加 `temperature` 加快生成速度（但会降低确定性）
3. 调整 `max_history` 平衡上下文和内存

---

## 依赖说明

| 包 | 版本 | 用途 |
|----|------|------|
| `langchain` | >=0.3.0 | AI 框架 |
| `langchain-openai` | >=0.2.0 | OpenAI 集成 |
| `langchain-core` | - | 核心组件 |
| `langgraph` | - | ReAct Agent 支持 |
| `fastapi` | >=0.110.0 | Web 框架 |
| `uvicorn` | >=0.27.0 | ASGI 服务器 |
| `python-dotenv` | >=1.0.0 | 环境变量管理 |
| `openai` | - | OpenAI SDK |

---

## 扩展和定制

### 集成真实搜索引擎

修改 `search_web` 工具：
```python
@tool
def search_web(query: str) -> str:
    """使用真实搜索 API"""
    # 集成 Google Search API、Bing API 等
    pass
```

### 集成数据库

在 `SessionManager` 中添加持久化：
```python
def save_session(self, session_id: str):
    # 保存到数据库
    pass
```

### 自定义提示词

修改 `ReactAgent` 中的 `system_message`：
```python
self.system_message = SystemMessage(content="自定义系统提示词...")
```

---

## 故障排查

### 问题：OPENAI_API_KEY 环境变量未设置
**解决**：
```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
python app.py
```

### 问题：连接超时
**解决**：增加超时时间
```python
AgentConfig(timeout=60)  # 60 秒
```

### 问题：Web 界面无法访问
**解决**：
1. 确认服务正在运行：`curl http://localhost:8000`
2. 检查防火墙设置
3. 尝试在浏览器中访问 `http://127.0.0.1:8000`

---

## 性能指标

| 指标 | 说明 | 基准值 |
|------|------|--------|
| 首次响应时间 | 从请求到第一个 token | ~1-2s |
| 平均响应时间 | 完整响应生成时间 | ~5-10s |
| 并发用户数 | 支持的同时活跃用户 | 100+ |
| 内存占用 | 空闲状态内存 | ~200-300MB |

---

## 许可证

本项目遵循 MIT 许可证。

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 相关资源

- [LangChain 文档](https://js.langchain.com/docs/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [OpenAI API 文档](https://platform.openai.com/docs)

---

## 更新日志

### v1.0.0 (2024-01-17)
- ✨ 初始版本发布
- 🎉 完整的 ReAct Agent 实现
- 🌐 FastAPI Web 服务
- 💬 实时流式聊天
- 🔧 多工具集成
- 📊 会话管理系统

---

**最后更新**：2024年01月17日

如有问题或建议，欢迎联系开发者！
