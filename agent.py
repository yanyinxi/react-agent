#!/usr/bin/env python3
"""
ReAct Agent - 使用LangChain实现的推理+行动Agent
支持工具调用，边推理边执行，带详细日志
"""

import os
import sys
import logging
import ast
import operator
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
from urllib.parse import urlparse
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent
from openai import APIError, RateLimitError, APITimeoutError
import math


# ============ 配置类 ============

@dataclass
class AgentConfig:
    """Agent 配置"""
    model: str = "claude-opus-4-5-20251101"
    temperature: float = 0
    max_tokens: int = 2000
    max_history: int = 50  # 最大对话历史条数
    timeout: int = 30  # API 超时秒数


# ============ 安全数学表达式计算 ============

class SafeMathEvaluator:
    """安全的数学表达式计算器，使用 AST 解析避免代码注入"""

    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    ALLOWED_FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'floor': math.floor,
        'ceil': math.ceil,
    }

    ALLOWED_CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
    }

    def evaluate(self, expression: str) -> float:
        """安全地计算数学表达式"""
        try:
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body)
        except (SyntaxError, TypeError, KeyError) as e:
            raise ValueError(f"无效的数学表达式: {e}")

    def _eval_node(self, node: ast.AST) -> float:
        """递归计算 AST 节点"""
        if isinstance(node, ast.Constant):  # 数字常量
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的常量类型: {type(node.value)}")

        elif isinstance(node, ast.Name):  # 变量名（常量如 pi, e）
            if node.id in self.ALLOWED_CONSTANTS:
                return self.ALLOWED_CONSTANTS[node.id]
            raise ValueError(f"未知的常量: {node.id}")

        elif isinstance(node, ast.BinOp):  # 二元运算
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.ALLOWED_OPERATORS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):  # 一元运算
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return self.ALLOWED_OPERATORS[op_type](operand)

        elif isinstance(node, ast.Call):  # 函数调用
            if not isinstance(node.func, ast.Name):
                raise ValueError("不支持复杂的函数调用")
            func_name = node.func.id
            if func_name not in self.ALLOWED_FUNCTIONS:
                raise ValueError(f"不支持的函数: {func_name}")
            args = [self._eval_node(arg) for arg in node.args]
            return self.ALLOWED_FUNCTIONS[func_name](*args)

        else:
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


# 全局安全计算器实例
_safe_math = SafeMathEvaluator()

# ============ 日志配置 ============

class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

# 配置日志
logger = logging.getLogger("ReActAgent")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter(
    fmt='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
))
logger.addHandler(handler)


# ============ 流式回调处理器 ============

class StreamingCallbackHandler(BaseCallbackHandler):
    """流式输出回调处理器"""
    
    def __init__(self):
        self.tokens = []
        self.current_tool = None
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        logger.info("🧠 LLM开始思考...")
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens.append(token)
        print(token, end="", flush=True)
    
    def on_llm_end(self, response, **kwargs):
        logger.info("✅ LLM思考完成")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.current_tool = tool_name
        logger.info(f"🔧 调用工具: {tool_name}")
        logger.info(f"   输入: {input_str}")
    
    def on_tool_end(self, output, **kwargs):
        logger.info(f"   输出: {output}")
        logger.info(f"✅ 工具 {self.current_tool} 执行完成")
    
    def on_tool_error(self, error, **kwargs):
        logger.error(f"❌ 工具执行错误: {error}")


# ============ 工具定义 ============

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持基本运算(+,-,*,/,**,%)和函数(sqrt,sin,cos,tan,log,abs,round等)，以及常量(pi,e)。示例: '2 + 2', 'sqrt(16)', '3.14 * 2**2'"""
    try:
        result = _safe_math.evaluate(expression)
        return f"计算结果: {result}"
    except ValueError as e:
        return f"计算错误: {str(e)}"
    except ZeroDivisionError:
        return "计算错误: 除数不能为零"
    except OverflowError:
        return "计算错误: 数值溢出"


@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')} (星期{['一','二','三','四','五','六','日'][now.weekday()]})"


@tool
def search_web(query: str) -> str:
    """搜索网络信息（模拟）。输入搜索关键词"""
    return f"搜索 '{query}' 的结果: 这是一个模拟的搜索结果。在实际应用中，您可以接入Google Search API、Bing API或其他搜索服务来获取真实结果。"


@tool  
def text_analyzer(text: str) -> str:
    """分析文本，返回字数、字符数等统计信息"""
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.split('\n'))
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    
    return f"""文本分析结果:
- 总字符数: {char_count}
- 单词/词组数: {word_count}
- 行数: {line_count}
- 中文字符数: {chinese_count}"""


# ============ ReAct Agent ============

class ReactAgent:
    """ReAct模式的Agent，支持工具调用和流式输出"""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        self.api_key = os.environ.get("OPENAI_API_KEY")

        # 验证环境变量
        self._validate_config()

        # 线程锁，保护对话历史
        self._lock = threading.Lock()

        logger.info("=" * 60)
        logger.info("🚀 初始化 ReAct Agent")
        logger.info(f"   模型: {self.config.model}")
        logger.info(f"   API: {self.base_url}")

        # 初始化LLM（带超时）
        self.llm = ChatOpenAI(
            model=self.config.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            streaming=True,
            request_timeout=self.config.timeout
        )

        # 定义工具列表
        self.tools = [
            calculator,
            get_current_time,
            search_web,
            text_analyzer
        ]

        logger.info(f"   工具: {[t.name for t in self.tools]}")

        # 系统提示
        self.system_message = SystemMessage(content="""你是一个智能助手，可以使用工具来帮助用户解决问题。
请用中文回答用户的问题。如果需要使用工具，请先说明你的思考过程，然后使用工具。""")

        # 创建ReAct Agent
        self.agent = create_react_agent(
            self.llm,
            self.tools
        )

        # 对话历史
        self.messages: List[Any] = []

        logger.info("✅ ReAct Agent 初始化完成")
        logger.info("=" * 60)

    def _validate_config(self) -> None:
        """验证环境变量配置"""
        if not self.base_url or not self.api_key:
            raise ValueError("请设置环境变量 OPENAI_BASE_URL 和 OPENAI_API_KEY")

        # 验证 API 密钥格式
        if len(self.api_key) < 10:
            raise ValueError("API 密钥格式无效（长度过短）")

        # 验证 Base URL 格式
        parsed = urlparse(self.base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Base URL 格式无效，应为完整 URL（如 https://api.example.com）")

    def _build_messages(self, user_input: str) -> List[Any]:
        """构建消息列表"""
        return [self.system_message] + self.messages + [HumanMessage(content=user_input)]

    def _add_to_history(self, user_input: str, output: str) -> None:
        """线程安全地添加对话到历史，并限制历史长度"""
        with self._lock:
            self.messages.append(HumanMessage(content=user_input))
            self.messages.append(AIMessage(content=output))
            # 限制历史长度，防止内存泄漏
            if len(self.messages) > self.config.max_history:
                self.messages = self.messages[-self.config.max_history:]

    def chat(self, user_input: str) -> Dict[str, Any]:
        """处理用户输入并返回响应"""
        logger.info("-" * 60)
        logger.info(f"👤 用户输入: {user_input}")
        logger.info("-" * 60)

        try:
            messages = self._build_messages(user_input)

            # 调用Agent
            logger.info("🤖 Agent开始处理...")
            result = self.agent.invoke({"messages": messages})

            # 提取最终回复
            final_message = result["messages"][-1]
            output = final_message.content if hasattr(final_message, 'content') else str(final_message)

            # 更新对话历史
            self._add_to_history(user_input, output)

            logger.info("-" * 60)
            logger.info(f"🤖 Agent回复: {output[:200]}{'...' if len(output) > 200 else ''}")
            logger.info("-" * 60)

            return {
                "output": output,
                "success": True
            }
        except RateLimitError:
            logger.error("❌ API 请求过于频繁")
            return {
                "output": "请求过于频繁，请稍后重试",
                "success": False
            }
        except APITimeoutError:
            logger.error("❌ API 请求超时")
            return {
                "output": "请求超时，请稍后重试",
                "success": False
            }
        except APIError as e:
            logger.error(f"❌ API 错误: {str(e)}")
            return {
                "output": f"API 错误: {str(e)}",
                "success": False
            }
        except Exception as e:
            logger.exception("❌ 处理请求时出错")
            return {
                "output": "服务暂时不可用，请稍后重试",
                "success": False
            }

    def chat_stream(self, user_input: str) -> Generator[str, None, None]:
        """流式处理用户输入"""
        logger.info("-" * 60)
        logger.info(f"👤 用户输入: {user_input}")
        logger.info("-" * 60)

        try:
            messages = self._build_messages(user_input)

            logger.info("🤖 Agent开始流式处理...")

            # 使用列表收集响应，避免 O(n²) 字符串拼接
            response_parts: List[str] = []

            for chunk in self.agent.stream({"messages": messages}):
                # 处理不同类型的chunk
                if "agent" in chunk:
                    agent_messages = chunk["agent"].get("messages", [])
                    for msg in agent_messages:
                        if hasattr(msg, 'content') and msg.content:
                            content = msg.content
                            response_parts.append(content)
                            yield content

                elif "tools" in chunk:
                    tool_messages = chunk["tools"].get("messages", [])
                    for msg in tool_messages:
                        tool_name = getattr(msg, 'name', 'unknown')
                        tool_content = msg.content if hasattr(msg, 'content') else str(msg)
                        logger.info(f"🔧 工具 {tool_name} 返回: {tool_content}")

            # 更新对话历史
            full_response = ''.join(response_parts)
            self._add_to_history(user_input, full_response)

            logger.info("-" * 60)
            logger.info(f"🤖 Agent回复完成，共 {len(full_response)} 字符")
            logger.info("-" * 60)

        except RateLimitError:
            logger.error("❌ API 请求过于频繁")
            yield "请求过于频繁，请稍后重试"
        except APITimeoutError:
            logger.error("❌ API 请求超时")
            yield "请求超时，请稍后重试"
        except APIError as e:
            logger.error(f"❌ API 错误: {str(e)}")
            yield f"API 错误: {str(e)}"
        except Exception as e:
            logger.exception("❌ 流式处理出错")
            yield "服务暂时不可用，请稍后重试"

    def reset(self) -> None:
        """重置对话历史"""
        with self._lock:
            self.messages = []
        logger.info("🔄 对话历史已重置")

    def get_tools_info(self) -> List[Dict[str, str]]:
        """获取可用工具信息"""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools
        ]


# 命令行测试
if __name__ == "__main__":
    print("=" * 60)
    print("ReAct Agent - 推理+行动智能助手")
    print("=" * 60)
    print("可用工具: calculator, get_current_time, search_web, text_analyzer")
    print("命令: 'quit' 退出, 'reset' 重置对话, 'tools' 查看工具")
    print("-" * 60)
    
    try:
        agent = ReactAgent()
    except ValueError as e:
        print(f"初始化失败: {e}")
        exit(1)
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("再见!")
                break
            if user_input.lower() == "reset":
                agent.reset()
                continue
            if user_input.lower() == "tools":
                print("\n可用工具:")
                for tool_info in agent.get_tools_info():
                    print(f"  - {tool_info['name']}: {tool_info['description']}")
                continue
            
            print("\nAgent: ", end="", flush=True)
            for chunk in agent.chat_stream(user_input):
                print(chunk, end="", flush=True)
            print()
            
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
