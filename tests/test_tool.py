"""测试 @tool 装饰器和注册系统"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.tool import tool, get_definition, get_all_definitions, execute, list_tools, clear


class TestToolDecorator:
    def setup_method(self):
        clear()

    def test_basic_registration(self):
        @tool(description="test tool")
        def my_tool(x: int, y: str) -> str:
            """My test tool doc"""
            return f"{x}:{y}"

        defs = get_all_definitions()
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "my_tool"
        assert defs[0]["function"]["description"] == "test tool"

    def test_custom_name(self):
        @tool(name="custom_name", description="custom desc")
        def some_func():
            pass

        assert "custom_name" in list_tools()
        assert "some_func" not in list_tools()

    def test_execute(self):
        @tool(description="add")
        def add(a: int, b: int) -> int:
            return a + b

        result = execute("add", {"a": 3, "b": 4})
        assert result == 7

    def test_execute_unknown_tool(self):
        result = execute("nonexistent", {})
        assert "Unknown tool" in result

    def test_execute_error(self):
        @tool(description="broken")
        def broken():
            raise ValueError("boom")

        result = execute("broken", {})
        assert "Tool Error" in result
        assert "boom" in result

    def test_schema_generation(self):
        @tool(description="test")
        def typed_func(name: str, count: int, price: float = 0.0) -> str:
            pass

        defn = get_definition("typed_func")
        params = defn["function"]["parameters"]
        props = params["properties"]

        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        assert props["price"]["type"] == "number"
        assert params["required"] == ["name", "count"]

    def test_clear(self):
        @tool(description="t")
        def t():
            pass

        assert len(list_tools()) == 1
        clear()
        assert len(list_tools()) == 0

    def test_multiple_tools(self):
        @tool(description="a")
        def tool_a():
            pass

        @tool(description="b")
        def tool_b():
            pass

        names = list_tools()
        assert "tool_a" in names
        assert "tool_b" in names
        assert len(names) == 2
