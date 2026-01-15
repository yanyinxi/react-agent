#!/usr/bin/env python3
"""
ReAct Agent - 使用LangChain实现的推理+行动Agent
支持工具调用，边推理边执行
"""

import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
import math
import datetime


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
    now = datetime.datetime.now()
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
    """ReAct模式的Agent，支持工具调用"""
    
    def __init__(self, model: str = "claude-opus-4-5-20251101"):
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.base_url or not self.api_key:
            raise ValueError("请设置环境变量 OPENAI_BASE_URL 和 OPENAI_API_KEY")
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=0,
            max_tokens=2000
        )
        
        # 定义工具列表
        self.tools = [
            calculator,
            get_current_time,
            search_web,
            text_analyzer
        ]
        
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
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """处理用户输入并返回响应"""
        try:
            # 构建消息列表
            messages = [self.system_message] + self.messages + [HumanMessage(content=user_input)]
            
            # 调用Agent
            result = self.agent.invoke({"messages": messages})
            
            # 提取最终回复
            final_message = result["messages"][-1]
            output = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            # 更新对话历史
            self.messages.append(HumanMessage(content=user_input))
            self.messages.append(AIMessage(content=output))
            
            return {
                "output": output,
                "success": True
            }
        except Exception as e:
            return {
                "output": f"处理请求时出错: {str(e)}",
                "success": False
            }
    
    def reset(self):
        """重置对话历史"""
        self.messages = []
    
    def get_tools_info(self) -> List[Dict[str, str]]:
        """获取可用工具信息"""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools
        ]


# 命令行测试
if __name__ == "__main__":
    print("=" * 50)
    print("ReAct Agent - 推理+行动智能助手")
    print("=" * 50)
    print("可用工具: calculator, get_current_time, search_web, text_analyzer")
    print("命令: 'quit' 退出, 'reset' 重置对话, 'tools' 查看工具")
    print("-" * 50)
    
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
                print("对话已重置")
                continue
            if user_input.lower() == "tools":
                print("\n可用工具:")
                for tool_info in agent.get_tools_info():
                    print(f"  - {tool_info['name']}: {tool_info['description']}")
                continue
            
            print("\nAgent思考中...")
            result = agent.chat(user_input)
            print(f"\nAgent: {result['output']}")
            
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
