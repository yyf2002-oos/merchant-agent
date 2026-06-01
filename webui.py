"""商家智能 Agent — Gradio Web 界面"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import io
import gradio as gr
from llm import check_llm, check_ollama, list_models, stream_chat
from orchestrator import MerchantOrchestrator
from config import LOG_LEVEL, LOG_FORMAT, RATE_LIMIT_ENABLED, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW, CACHE_ENABLED, LLM_PROVIDER

# ── 日志配置 ──
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

orch = MerchantOrchestrator()

# ── 速率限制（IP 滑动窗口） ──
_rate_limit_store: dict[str, list[float]] = {}


def _check_rate_limit(ip: str) -> bool:
    if not RATE_LIMIT_ENABLED:
        return True
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    if ip not in _rate_limit_store:
        _rate_limit_store[ip] = []
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if t > cutoff]
    if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
        logger.warning(f"速率限制触发: IP={ip}")
        return False
    _rate_limit_store[ip].append(now)
    return True


def _check_llm():
    """根据 LLM_PROVIDER 自动检查 DeepSeek 或 Ollama"""
    provider = LLM_PROVIDER
    if provider == "deepseek":
        ok, msg = check_llm()
        if not ok:
            return False, f"❌ **DeepSeek 不可用** — {msg}"
        return True, ""
    else:
        if not check_ollama():
            return False, "❌ **Ollama 未运行！** 请先执行 `ollama serve` 启动服务。"
        return True, ""


def check_env():
    ok, msg = check_llm()
    models = list_models() if ok else []
    return ok, models, msg


def agent_chat(message, history):
    """智能对话 Tab — 带限流"""
    ok, err = _check_llm()
    if not ok:
        yield err
        return
    if not message.strip():
        yield "请输入你的问题"
        return
    if not _check_rate_limit("webui_chat"):
        yield "⚠️ **请求过于频繁，请稍后再试**（每分钟限制 30 次）"
        return

    try:
        logger.info(f"WebUI chat: {message[:60]}")
        yield "🤔 正在分析你的问题..."
        response = orch.smart_chat(message)
        yield response
    except Exception as e:
        logger.error(f"chat 异常: {e}", exc_info=True)
        yield f"❌ **处理出错**: {e}"


def run_selector(category, budget, audience):
    ok, err = _check_llm()
    if not ok:
        return err, None
    if not category.strip():
        return "请输入品类方向", None

    try:
        logger.info(f"WebUI selector: {category}")
        result = orch.run_agent("selector", category, budget=budget, target_audience=audience)
        report = result.get("report", "生成失败")
        recs = result.get("recommendations", [])
        rec_text = "📊 数据来源：淘宝搜索下拉词（真实用户搜索需求）\n\n"
        for r in recs:
            price = r.get("avg_price", 0)
            cost = r.get("cost", 0)
            margin = round((price - cost) / price * 100, 1) if price > 0 else 0
            rec_text += f"• {r.get('name', '')} — ¥{price} | 成本¥{cost} | 毛利率{margin}%\n"
        return report, rec_text if rec_text else None
    except Exception as e:
        logger.error(f"selector 异常: {e}", exc_info=True)
        return f"❌ **处理出错**: {e}", None


def run_lister(name, category, features, price, cost):
    ok, err = _check_llm()
    if not ok:
        return err
    if not name.strip():
        return "请输入商品名称"

    try:
        logger.info(f"WebUI lister: {name[:30]}")
        info = {
            "name": name,
            "category": category,
            "features": features,
            "target_price": price,
            "cost": cost,
        }
        result = orch.run_agent("lister", info)
        content = result.get("listing_content", "生成失败")
        saved = result.get("saved_to", "")
        if saved:
            content += f"\n\n---\n📁 已保存到: `{saved}`"
        return content
    except Exception as e:
        logger.error(f"lister 异常: {e}", exc_info=True)
        return f"❌ **处理出错**: {e}"


def run_sourcing(name, category, target_price, expected_sales, budget):
    ok, err = _check_llm()
    if not ok:
        return err
    if not name.strip() and not category.strip():
        return "请输入商品名称或品类"

    try:
        logger.info(f"WebUI sourcing: {name[:30] or category[:30]}")
        info = {
            "name": name,
            "category": category,
            "target_price": target_price,
            "expected_sales": expected_sales,
            "budget": budget,
        }
        result = orch.run_agent("sourcing", info)
        report = result.get("report", "生成失败")
        tool_calls = result.get("tool_calls", 0)
        footer = f"\n\n---\n🤖 AI 自主调用了 {tool_calls} 次工具分析"
        return report + footer
    except Exception as e:
        logger.error(f"sourcing 异常: {e}", exc_info=True)
        return f"❌ **处理出错**: {e}"


def run_service(query, product_context):
    ok, err = _check_llm()
    if not ok:
        return err
    if not query.strip():
        return "请输入顾客问题"

    try:
        logger.info(f"WebUI service: {query[:40]}")
        result = orch.run_agent("service", query, product_context=product_context)
        return result.get("answer", "生成失败")
    except Exception as e:
        logger.error(f"service 异常: {e}", exc_info=True)
        return f"❌ **处理出错**: {e}"


def run_analyst(data, cost, price, volume):
    ok, err = _check_llm()
    if not ok:
        return err, None
    if not data.strip():
        return "请输入经营数据", None

    try:
        logger.info(f"WebUI analyst: {data[:40]}")
        input_data = data
        pricing_data = {}
        if cost:
            pricing_data["cost"] = cost
        if price:
            pricing_data["price"] = price
        if volume:
            pricing_data["volume"] = volume

        result = orch.run_agent("analyst", input_data if not pricing_data else {**pricing_data, "data": data})
        report = result.get("report", "生成失败")
        pricing = result.get("pricing_advice")
        pricing_text = ""
        if pricing:
            pa = pricing.get("profit_analysis", {})
            pricing_text = f"利润: ¥{pa.get('profit', 0)} | 利润率: {pa.get('margin', 0)}%"
            ps = pricing.get("price_suggestion", {})
            pricing_text += f"\n建议售价: ¥{ps.get('competitive_price', 0)} (常规) / ¥{ps.get('premium_price', 0)} (溢价)"
        return report, pricing_text if pricing_text else None
    except Exception as e:
        logger.error(f"analyst 异常: {e}", exc_info=True)
        return f"❌ **处理出错**: {e}", None


def run_workflow(category, budget, audience):
    ok, err = _check_llm()
    if not ok:
        yield err
        return
    if not category.strip():
        yield "请输入品类方向"
        return

    try:
        logger.info(f"WebUI workflow: {category}")
        yield "🚀 开始执行完整工作流...\n\n"

        yield "📋 Step 1/4: 选品分析中...\n"
        results = orch.run_full_workflow(category, budget, audience)

        output = f"# 🛒 完整工作流报告：{category}\n\n"

        selector = results.get("selector", {})
        output += f"## 📋 选品分析\n{selector.get('report', 'N/A')}\n\n"
        yield output

        lister = results.get("lister", {})
        output += f"## 📝 上架素材\n{lister.get('listing_content', 'N/A')}\n\n"
        yield output

        service = results.get("service", {})
        output += f"## 💬 客服应答\n{service.get('answer', 'N/A')}\n\n"
        yield output

        analyst = results.get("analyst", {})
        output += f"## 📊 运营建议\n{analyst.get('report', 'N/A')}\n\n"
        yield output

        output += "---\n✅ **全部完成！**"
        yield output
    except Exception as e:
        logger.error(f"workflow 异常: {e}", exc_info=True)
        yield f"❌ **工作流出错**: {e}"


def parse_csv(file):
    """解析上传的 CSV 文件"""
    if file is None:
        return None, "请上传 CSV 文件"
    try:
        content = file.decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        products = []
        for row in reader:
            name = row.get("名称", "") or row.get("name", "") or ""
            category = row.get("品类", "") or row.get("category", "") or ""
            features = row.get("卖点", "") or row.get("features", "") or ""
            price = row.get("售价", "") or row.get("price", "") or ""
            cost = row.get("成本", "") or row.get("cost", "") or ""
            if name:
                products.append({
                    "name": name,
                    "category": category,
                    "features": features,
                    "target_price": price,
                    "cost": cost,
                })
        if not products:
            return None, "CSV 中未找到有效商品数据。请确保包含列：名称,品类,卖点,售价,成本"
        return products, f"成功解析 {len(products)} 个商品"
    except Exception as e:
        return None, f"解析失败: {str(e)}"


def run_batch_listing(file):
    """批量上架"""
    ok, err = _check_llm()
    if not ok:
        yield f"❌ {err}", None
        return

    products, msg = parse_csv(file)
    if not products:
        yield msg, None
        return

    yield f"📦 开始批量生成 {len(products)} 个商品...\n", None

    results = orch.batch_listing(products)

    # 生成输出文本
    output = f"# ✅ 批量上架完成（{len(results)} 个商品）\n\n"
    for r in results:
        output += f"---\n### {r['index']}. {r['name']}\n{r['content']}\n\n"
    yield output, results


def export_batch_csv(results):
    """导出批量上架结果为 CSV"""
    if not results:
        return None
    output = io.StringIO()
    output.write("商品名称,标题方案1,标题方案2,标题方案3,核心卖点\n")
    for r in results:
        # 简单解析 — 从内容里提取标题
        name = r.get("name", "")
        content = r.get("content", "")
        lines = content.split("\n")
        titles = [l for l in lines if "方案" in l and "：" in l]
        title1 = titles[0].split("：")[-1].strip() if len(titles) > 0 else ""
        title2 = titles[1].split("：")[-1].strip() if len(titles) > 1 else ""
        title3 = titles[2].split("：")[-1].strip() if len(titles) > 2 else ""
        output.write(f"{name},{title1},{title2},{title3},\n")
    return output.getvalue()


# ===== 构建 Gradio 界面 =====

def build_app():
    ok, models, detail = check_env()
    provider_label = LLM_PROVIDER.upper()
    status = f"✅ {provider_label}" if ok else f"❌ {provider_label}"
    model_list = ", ".join(models[:3]) if models else "无"

    with gr.Blocks(title="商家智能运营 Agent") as app:
        gr.Markdown("""# 🛒 商家智能运营 Agent
        👉 选品 → 上架 → 客服 → 运营分析 — AI 全流程自动完成
        """)

        with gr.Row():
            cache_status = "✅" if CACHE_ENABLED else "❌"
            rate_status = "✅" if RATE_LIMIT_ENABLED else "❌"
            status_md = gr.Markdown(
                f"**{status}** | 模型: {model_list} "
                f"| 缓存: {cache_status} | 限流: {rate_status}"
            )
            refresh_btn = gr.Button("🔄 刷新状态", size="sm", scale=0)

        def refresh_status():
            ok, models, detail = check_env()
            s = f"✅ {LLM_PROVIDER.upper()}" if ok else f"❌ {LLM_PROVIDER.upper()}"
            ms = ", ".join(models[:3]) if models else "无"
            cs = "✅" if CACHE_ENABLED else "❌"
            rs = "✅" if RATE_LIMIT_ENABLED else "❌"
            return f"**{s}** | 模型: {ms} | 缓存: {cs} | 限流: {rs}"

        refresh_btn.click(refresh_status, outputs=status_md)

        with gr.Tabs():
            # ===== Tab 1: 智能对话 =====
            with gr.TabItem("💬 智能对话"):
                gr.Markdown("随便问，自动路由到对应的 Agent")
                gr.ChatInterface(
                    agent_chat,
                    title="智能电商助手",
                    description="输入任何电商相关问题",
                )

            # ===== Tab 2: 选品 =====
            with gr.TabItem("📋 选品分析"):
                gr.Markdown("输入品类方向 → 自动抓取**淘宝真实搜索数据** → AI 分析给出选品推荐")
                with gr.Row():
                    with gr.Column():
                        category_in = gr.Textbox(label="品类方向", placeholder="如: 学生文具、宠物用品、数码配件")
                        budget_in = gr.Textbox(label="启动预算（可选）", placeholder="如: 10000")
                        audience_in = gr.Textbox(label="目标人群（可选）", placeholder="如: 大学生、上班族")
                        btn_selector = gr.Button("开始分析", variant="primary")
                    with gr.Column():
                        selector_out = gr.Markdown(label="分析报告")
                        recs_out = gr.Textbox(label="推荐商品评分", lines=5)
                btn_selector.click(run_selector, [category_in, budget_in, audience_in], [selector_out, recs_out])

            # ===== Tab 3: 上架 =====
            with gr.TabItem("📝 上架素材"):
                gr.Markdown("输入商品信息，AI 生成完整上架素材包")
                with gr.Row():
                    with gr.Column():
                        name_in = gr.Textbox(label="商品名称", placeholder="如: 智能保温杯")
                        cat_in = gr.Textbox(label="所属品类", placeholder="如: 家居日用")
                        feat_in = gr.Textbox(label="核心卖点（逗号分隔）", placeholder="如: 24小时保温,316不锈钢,500ml大容量")
                        price_in = gr.Textbox(label="目标售价", placeholder="如: 89")
                        cost_in = gr.Textbox(label="成本价", placeholder="如: 35")
                        btn_lister = gr.Button("生成素材", variant="primary")
                    with gr.Column():
                        lister_out = gr.Markdown(label="上架素材")
                btn_lister.click(run_lister, [name_in, cat_in, feat_in, price_in, cost_in], lister_out)

            # ===== Tab 4: 批量上架 =====
            with gr.TabItem("📦 批量上架"):
                gr.Markdown("上传 CSV 批量生成上架素材。CSV 需包含列：**名称,品类,卖点,售价,成本**")
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="上传 CSV 文件", file_types=[".csv"])
                        btn_batch = gr.Button("🚀 批量生成", variant="primary")
                    with gr.Column():
                        batch_out = gr.Markdown(label="生成结果")
                state = gr.State()
                download_file = gr.File(label="下载 CSV 结果")

                btn_batch.click(run_batch_listing, file_input, [batch_out, state])
                state.change(export_batch_csv, state, download_file)

            # ===== Tab 5: 货源 =====
            with gr.TabItem("🏭 一手货源"):
                gr.Markdown("""
                输入商品信息，AI 分析**一手工厂货源**和**地区产业集群优势」。

                **核心功能：**
                - 告诉你哪个县/镇是这类品的**工厂集群**
                - 帮你**区分一手工厂和贸易商**
                - 给出**可执行的采购行动计划**
                """)
                with gr.Row():
                    with gr.Column():
                        src_name = gr.Textbox(label="商品名称", placeholder="如: 迷你手持小风扇")
                        src_cat = gr.Textbox(label="所属品类", placeholder="如: 电风扇")
                        src_price = gr.Textbox(label="目标售价（可选）", placeholder="如: 39")
                        src_sales = gr.Textbox(label="预期月销量（可选）", placeholder="如: 500")
                        src_budget = gr.Textbox(label="采购预算（可选）", placeholder="如: 5000")
                        btn_sourcing = gr.Button("🔍 分析一手货源", variant="primary")
                    with gr.Column():
                        sourcing_out = gr.Markdown(label="一手货源分析报告")
                btn_sourcing.click(run_sourcing, [src_name, src_cat, src_price, src_sales, src_budget], sourcing_out)

            # ===== Tab 6: 客服 =====
            with gr.TabItem("💬 客服应答"):
                gr.Markdown("输入顾客问题，AI 给出客服回复（自动检索 FAQ 知识库）")
                with gr.Row():
                    with gr.Column():
                        query_in = gr.Textbox(label="顾客问题", placeholder="如: 什么时候发货？", lines=3)
                        ctx_in = gr.Textbox(label="商品信息（可选）", placeholder="当前商品的名称/规格信息", lines=2)
                        btn_service = gr.Button("生成回复", variant="primary")
                    with gr.Column():
                        service_out = gr.Markdown(label="客服回复")
                btn_service.click(run_service, [query_in, ctx_in], service_out)

            # ===== Tab 7: 运营分析 =====
            with gr.TabItem("📊 运营分析"):
                gr.Markdown("输入经营数据，AI 给出分析报告和运营建议")
                with gr.Row():
                    with gr.Column():
                        data_in = gr.Textbox(label="经营数据", placeholder="如: 店铺A上周销售数据：商品卖出200件，销售额17800元...", lines=5)
                        with gr.Row():
                            cost_an = gr.Number(label="成本价", value=0)
                            price_an = gr.Number(label="售价", value=0)
                            vol_an = gr.Number(label="销量", value=0)
                        btn_analyst = gr.Button("开始分析", variant="primary")
                    with gr.Column():
                        analyst_out = gr.Markdown(label="分析报告")
                        pricing_out = gr.Textbox(label="定价建议", lines=3)
                btn_analyst.click(run_analyst, [data_in, cost_an, price_an, vol_an], [analyst_out, pricing_out])

            # ===== Tab 8: 一键工作流 =====
            with gr.TabItem("🚀 一键完整工作流"):
                gr.Markdown("输入品类，AI 自动完成 选品→上架→客服→分析 全流程")
                with gr.Row():
                    with gr.Column():
                        wf_category = gr.Textbox(label="品类方向", placeholder="如: 宠物用品")
                        wf_budget = gr.Textbox(label="预算（可选）", placeholder="如: 5000")
                        wf_audience = gr.Textbox(label="目标人群（可选）", placeholder="如: 养猫人群")
                        btn_workflow = gr.Button("🚀 一键执行", variant="primary")
                    with gr.Column():
                        wf_out = gr.Markdown(label="工作流报告")
                btn_workflow.click(run_workflow, [wf_category, wf_budget, wf_audience], wf_out)

        gr.Markdown("---\n💡 提示：所有 Agent 使用本地 Ollama 模型运行，无需网络 API 调用。")

    return app


def main():
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme="soft",
    )


if __name__ == "__main__":
    main()
