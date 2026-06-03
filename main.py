"""商家智能 Agent — CLI 入口（带会话上下文管理）"""

import sys
import os
import uuid
from datetime import datetime

# Windows GBK 终端兼容：设置 UTF-8 编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

# 确保项目在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from config import LLM_PROVIDER
from llm import check_llm, check_ollama, list_models
from orchestrator import MerchantOrchestrator

console = Console()
orch = MerchantOrchestrator()

# 全局 Session ID（CLI 启动时生成一次，整个 CLI 会话共享）
CLI_SESSION_ID = f"cli_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def print_header():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]🛒 商家智能运营 Agent[/bold cyan]\n"
        "[yellow]AI 驱动的选品 → 上架 → 客服 → 运营 全流程[/yellow]\n"
        f"[dim]会话: {CLI_SESSION_ID[:16]}... | 所有对话带上下文记忆[/dim]",
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
    provider_label = LLM_PROVIDER.upper()
    console.print(f"[yellow]正在检查 {provider_label} 状态...[/yellow]")

    ok, msg = check_llm()
    if not ok:
        console.print(f"[red]❌ {provider_label} 不可用：{msg}[/red]")
        return

    console.print(f"[green]✅ {provider_label} 已连接[/green]")
    if LLM_PROVIDER == "ollama":
        models = list_models()
        console.print(f"   可用模型: {', '.join(models[:4])}")
    console.print("[green]   混合路由：选品/运营/货源 → DeepSeek | 上架/客服 → 本地 Ollama[/green]")

    print_agents()

    while True:
        console.print("\n[bold cyan]选择模式:[/bold cyan]")
        console.print("1. 单独 Agent 任务")
        console.print("2. 一键完整工作流（选品→上架→客服→分析）")
        console.print("3. 智能对话（自动路由 + 多轮上下文记忆）")
        console.print("4. 货源筛选（1688供应商分析）")
        console.print("5. 📊 商品价格库管理（录入/搜索/统计）")
        console.print("0. 退出")
        choice = console.input("\n[bold]请输入: [/bold]").strip()

        if choice == "0":
            console.print("[yellow]再见！[/yellow]")
            break

        elif choice == "5":
            _price_db_menu()

        elif choice == "1":
            print_agents()
            try:
                idx = int(console.input("输入 Agent 编号: ")) - 1
                agents = orch.list_agents()
                if 0 <= idx < len(agents):
                    agent_key = agents[idx]["key"]
                    prompt = console.input("请输入任务描述: ")
                    console.print("[yellow]正在处理，请稍候...[/yellow]")
                    result = orch.run_agent(agent_key, prompt, session_id=CLI_SESSION_ID)
                    for k, v in result.items():
                        if k not in ("agent", "session_id"):
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

            results = orch.run_full_workflow(category, budget, audience, session_id=CLI_SESSION_ID)

            for step_key in ["selector", "lister", "service", "analyst"]:
                step = results.get(step_key, {})
                agent_name = step.get("agent", step_key)
                console.print(f"\n[bold green]=== {agent_name} ===[/bold green]")
                for k, v in step.items():
                    if k not in ("agent", "session_id") and v:
                        console.print(Panel(str(v)[:1500], title=k))

        elif choice == "3":
            console.print("[dim]（按 Ctrl+C 返回主菜单）[/dim]")
            while True:
                try:
                    query = console.input("\n[bold cyan]你: [/bold cyan]")
                    if not query.strip():
                        continue
                    console.print("[yellow]正在思考...[/yellow]")
                    response = orch.smart_chat(query, session_id=CLI_SESSION_ID)
                    console.print(Panel(response, title="回复"))
                except KeyboardInterrupt:
                    console.print("\n[yellow]返回主菜单[/yellow]")
                    break

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
            result = orch.run_agent("sourcing", info, session_id=CLI_SESSION_ID)
            console.print(Panel(result.get("report", "生成失败")[:3000], title="货源分析"))
            search_url = result.get("search_url", "")
            if search_url:
                console.print(f"\n[blue]🔍 1688搜索链接: {search_url}[/blue]")

    console.print("[green]谢谢使用！[/green]")


def _price_db_menu():
    """商品价格库管理"""
    from tools.product_manager import add_product, search_products, delete_product, get_stats, list_categories

    while True:
        console.print("\n[bold cyan]=== 商品价格库管理 ===[/bold cyan]")
        stats = get_stats()
        console.print(f"[dim]当前库: {stats['total']} 个商品, {stats['categories']} 个品类")
        if stats['total'] > 0:
            console.print(f"  价格范围: ¥{stats['min_price']} ~ ¥{stats['max_price']} | 均价: ¥{stats['avg_price']}[/dim]")

        console.print("1. 录入商品")
        console.print("2. 搜索商品")
        console.print("3. 查看统计")
        console.print("4. 删除记录")
        console.print("0. 返回主菜单")
        choice = console.input("\n[bold]请输入: [/bold]").strip()

        if choice == "0":
            break

        elif choice == "1":
            name = console.input("商品名称: ")
            try:
                price = float(console.input("价格: "))
            except ValueError:
                console.print("[red]价格必须为数字[/red]")
                continue
            category = console.input("品类（可选）: ")
            platform = console.input("来源平台（可选，如 1688/淘宝/拼多多）: ")
            note = console.input("备注（可选）: ")
            rec = add_product(name, price, category, platform, note)
            console.print(f"[green]✅ 已录入：{rec['name']} ¥{rec['price']}[/green]")

        elif choice == "2":
            keyword = console.input("关键词（可选，直接回车跳过）: ")
            category = console.input("品类筛选（可选）: ")
            max_p = console.input("最高价格（可选）: ")
            min_p = console.input("最低价格（可选）: ")
            max_price = float(max_p) if max_p else 0
            min_price = float(min_p) if min_p else 0
            results = search_products(keyword, category, max_price, min_price)
            if not results:
                console.print("[yellow]未找到匹配记录[/yellow]")
            else:
                table = Table(title=f"找到 {len(results)} 个商品")
                table.add_column("ID", style="cyan")
                table.add_column("名称")
                table.add_column("价格", style="green")
                table.add_column("品类")
                table.add_column("平台", style="blue")
                table.add_column("备注")
                for r in results:
                    table.add_row(
                        str(r.get("id", "")),
                        r.get("name", "")[:20],
                        f"¥{r.get('price', 0)}",
                        r.get("category", "")[:10],
                        r.get("platform", "")[:10],
                        r.get("note", "")[:20],
                    )
                console.print(table)

        elif choice == "3":
            stats = get_stats()
            if stats["total"] == 0:
                console.print("[yellow]商品库为空，请先录入[/yellow]")
            else:
                console.print(f"[green]总商品数: {stats['total']}[/green]")
                console.print(f"[green]品类数: {stats['categories']}[/green]")
                console.print(f"[green]价格区间: ¥{stats['min_price']} ~ ¥{stats['max_price']}[/green]")
                console.print(f"[green]均价: ¥{stats['avg_price']}[/green]")
                cats = list_categories()
                if cats:
                    console.print(f"[dim]已有品类: {', '.join(cats)}[/dim]")

        elif choice == "4":
            try:
                pid = int(console.input("输入要删除的商品 ID: "))
                if delete_product(pid):
                    console.print("[green]已删除[/green]")
                else:
                    console.print("[red]未找到该 ID[/red]")
            except ValueError:
                console.print("[red]请输入数字[/red]")
        else:
            console.print("[red]无效选择[/red]")


def main():
    try:
        run_interactive()
    except KeyboardInterrupt:
        console.print("\n[yellow]已退出[/yellow]")
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")


if __name__ == "__main__":
    main()
