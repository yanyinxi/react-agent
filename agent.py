#!/usr/bin/env python3
"""
ReAct Agent - 使用LangChain实现的推理+行动Agent
支持工具调用，边推理边执行，带详细日志
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent
import math

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
    """计算数学表达式。输入应该是一个有效的Python数学表达式，如 '2 + 2' 或 'math.sqrt(16)'"""
    try:
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "len": len,
            "math": math, "sqrt": math.sqrt, "sin": math.sin,
            "cos": math.cos, "tan": math.tan, "log": math.log,
            "pi": math.pi, "e": math.e
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


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
    
    def __init__(self, model: str = "claude-opus-4-5-20251101"):
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.base_url or not self.api_key:
            raise ValueError("请设置环境变量 OPENAI_BASE_URL 和 OPENAI_API_KEY")
        
        logger.info("=" * 60)
        logger.info("🚀 初始化 ReAct Agent")
        logger.info(f"   模型: {model}")
        logger.info(f"   API: {self.base_url}")
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=0,
            max_tokens=2000,
            streaming=True
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
        self.messages = []
        
        logger.info("✅ ReAct Agent 初始化完成")
        logger.info("=" * 60)
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """处理用户输入并返回响应"""
        logger.info("-" * 60)
        logger.info(f"👤 用户输入: {user_input}")
        logger.info("-" * 60)
        
        try:
            # 构建消息列表
            messages = [self.system_message] + self.messages + [HumanMessage(content=user_input)]
            
            # 调用Agent
            logger.info("🤖 Agent开始处理...")
            result = self.agent.invoke({"messages": messages})
            
            # 提取最终回复
            final_message = result["messages"][-1]
            output = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            # 更新对话历史
            self.messages.append(HumanMessage(content=user_input))
            self.messages.append(AIMessage(content=output))
            
            logger.info("-" * 60)
            logger.info(f"🤖 Agent回复: {output[:200]}{'...' if len(output) > 200 else ''}")
            logger.info("-" * 60)
            
            return {
                "output": output,
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ 处理请求时出错: {str(e)}")
            return {
                "output": f"处理请求时出错: {str(e)}",
                "success": False
            }
    
    def chat_stream(self, user_input: str) -> Generator[str, None, None]:
        """流式处理用户输入"""
        logger.info("-" * 60)
        logger.info(f"👤 用户输入: {user_input}")
        logger.info("-" * 60)
        
        try:
            messages = [self.system_message] + self.messages + [HumanMessage(content=user_input)]
            
            logger.info("🤖 Agent开始流式处理...")
            
            full_response = ""
            
            for chunk in self.agent.stream({"messages": messages}):
                # 处理不同类型的chunk
                if "agent" in chunk:
                    agent_messages = chunk["agent"].get("messages", [])
                    for msg in agent_messages:
                        if hasattr(msg, 'content') and msg.content:
                            content = msg.content
                            full_response += content
                            yield content
                
                elif "tools" in chunk:
                    tool_messages = chunk["tools"].get("messages", [])
                    for msg in tool_messages:
                        tool_name = getattr(msg, 'name', 'unknown')
                        tool_content = msg.content if hasattr(msg, 'content') else str(msg)
                        logger.info(f"🔧 工具 {tool_name} 返回: {tool_content}")
            
            # 更新对话历史
            self.messages.append(HumanMessage(content=user_input))
            self.messages.append(AIMessage(content=full_response))
            
            logger.info("-" * 60)
            logger.info(f"🤖 Agent回复完成，共 {len(full_response)} 字符")
            logger.info("-" * 60)
            
        except Exception as e:
            logger.error(f"❌ 流式处理出错: {str(e)}")
            yield f"处理请求时出错: {str(e)}"
    
    def reset(self):
        """重置对话历史"""
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
