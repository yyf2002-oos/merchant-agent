"""商家智能 Agent — CLI 入口"""

import sys
import os

# 确保项目在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from llm import check_ollama, list_models
from orchestrator import MerchantOrchestrator

console = Console()
orch = MerchantOrchestrator()


def print_header():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]🛒 商家智能运营 Agent[/bold cyan]\n"
        "[yellow]AI 驱动的选品 → 上架 → 客服 → 运营 全流程[/yellow]",
        border_style="cyan",
    ))


def print_agents():
    table = Table(title="可用 Agent")
    table.add_column("编号", style="cyan")
    table.add_column("名称", style="green")
    table.add_column("说明")
    for i, a in enumerate(orch.list_agents(), 1):
        table.add_row(str(i), a["name"], a["desc"])
    console.print(table)


def run_interactive():
    print_header()
    console.print("[yellow]正在检查 Ollama 状态...[/yellow]")

    if not check_ollama():
        console.print("[red]❌ Ollama 未运行！请先启动 Ollama。[/red]")
        return

    models = list_models()
    console.print(f"[green]✅ Ollama 已连接[/green] | 可用模型: {', '.join(models[:4])}")

    print_agents()

    while True:
        console.print("\n[bold cyan]选择模式:[/bold cyan]")
        console.print("1. 单独 Agent 任务")
        console.print("2. 一键完整工作流（选品→上架→客服→分析）")
        console.print("3. 智能对话（自动路由到对应 Agent）")
        console.print("4. 货源筛选（1688供应商分析）")
        console.print("0. 退出")
        choice = console.input("\n[bold]请输入: [/bold]").strip()

        if choice == "0":
            console.print("[yellow]再见！[/yellow]")
            break

        elif choice == "1":
            print_agents()
            try:
                idx = int(console.input("输入 Agent 编号: ")) - 1
                agents = orch.list_agents()
                if 0 <= idx < len(agents):
                    agent_key = agents[idx]["key"]
                    prompt = console.input("请输入任务描述: ")
                    console.print("[yellow]正在处理，请稍候...[/yellow]")
                    result = orch.run_agent(agent_key, prompt)
                    for k, v in result.items():
                        if k != "agent":
                            console.print(Panel(str(v)[:2000], title=k))
                else:
                    console.print("[red]无效编号[/red]")
            except ValueError:
                console.print("[red]请输入数字[/red]")

        elif choice == "2":
            console.print("\n[bold]=== 一键完整工作流 ===[/bold]")
            category = console.input("品类方向: ")
            budget = console.input("预算（可选，直接回车跳过）: ")
            audience = console.input("目标人群（可选）: ")
            console.print("[yellow]正在执行完整工作流，这可能需要几分钟...[/yellow]")

            results = orch.run_full_workflow(category, budget, audience)

            for step_key in ["selector", "lister", "service", "analyst"]:
                step = results.get(step_key, {})
                agent_name = step.get("agent", step_key)
                console.print(f"\n[bold green]=== {agent_name} ===[/bold green]")
                for k, v in step.items():
                    if k not in ("agent",) and v:
                        console.print(Panel(str(v)[:1500], title=k))

        elif choice == "3":
            query = console.input("请输入你的问题: ")
            console.print("[yellow]正在思考...[/yellow]")
            response = orch.smart_chat(query)
            console.print(Panel(response, title="回复"))

        elif choice == "4":
            console.print("\n[bold]=== 货源筛选（1688供应商分析）=== [/bold]")
            name = console.input("商品名称: ")
            category = console.input("所属品类（可选）: ")
            target_price = console.input("目标售价（可选）: ")
            expected_sales = console.input("预期月销量（可选）: ")
            budget = console.input("采购预算（可选）: ")
            console.print("[yellow]正在分析供应商...[/yellow]")

            info = {"name": name, "category": category, "target_price": target_price,
                    "expected_sales": expected_sales, "budget": budget}
            result = orch.run_agent("sourcing", info)
            console.print(Panel(result.get("report", "生成失败")[:3000], title="货源分析"))
            search_url = result.get("search_url", "")
            if search_url:
                console.print(f"\n[blue]🔍 1688搜索链接: {search_url}[/blue]")

    console.print("[green]谢谢使用！[/green]")


def main():
    try:
        run_interactive()
    except KeyboardInterrupt:
        console.print("\n[yellow]已退出[/yellow]")
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")


if __name__ == "__main__":
    main()
