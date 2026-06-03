"""商家智能 Agent — Gradio Web 界面（知识库管理 + 系统监控 + 用户登录）"""

import sys
import os
import time
import logging
import uuid
import json
import hashlib
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import io
import gradio as gr
from llm import check_llm, check_ollama, list_models, stream_chat
from orchestrator import MerchantOrchestrator
from knowledge.data_manager import (
    get_all_faqs, add_faq, update_faq, delete_faq,
    get_price_library, add_price_subcategory, delete_price_subcategory,
    get_suppliers, add_supplier, delete_supplier_product,
    get_all_templates, get_all_categories,
)
from monitor import get_stats, get_recent_calls, get_daily_stats
from config import LOG_LEVEL, LOG_FORMAT, RATE_LIMIT_ENABLED, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW, CACHE_ENABLED, LLM_PROVIDER, WEB_PORT, ADMIN_USER, ADMIN_PASS

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

orch = MerchantOrchestrator()

_rate_limit_store: dict[str, list[float]] = {}

USERS_DB = os.path.join(os.path.dirname(__file__), "agent_memory.db")


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _init_users():
    """Initialize users table with hashed password storage"""
    import sqlite3
    conn = sqlite3.connect(USERS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create default admin account if not exists
    try:
        conn.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
                     (ADMIN_USER, _hash_password(ADMIN_PASS)))
        conn.commit()
    except sqlite3.Error as e:
        logger.warning(f"创建默认用户失败: {e}")
    conn.close()


_init_users()


def check_login(username: str, password: str) -> bool:
    """Gradio auth callback with hashed password verification"""
    import sqlite3
    try:
        conn = sqlite3.connect(USERS_DB)
        row = conn.execute("SELECT 1 FROM users WHERE username=? AND password=?",
                           (username, _hash_password(password))).fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False


def _check_rate_limit(ip: str) -> bool:
    if not RATE_LIMIT_ENABLED:
        return True
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    if ip not in _rate_limit_store:
        _rate_limit_store[ip] = []
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if t > cutoff]
    if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_limit_store[ip].append(now)
    return True


def _check_llm():
    provider = LLM_PROVIDER
    if provider == "deepseek":
        ok, msg = check_llm()
        if not ok:
            return False, f"❌ DeepSeek 不可用 — {msg}"
        return True, ""
    else:
        if not check_ollama():
            return False, "❌ Ollama 未运行！请先执行 ollama serve"
        return True, ""


def check_env():
    ok, msg = check_llm()
    models = list_models() if ok else []
    return ok, models, msg


def _get_session_id(request: gr.Request) -> str:
    """基于用户名+时间戳生成 session"""
    username = getattr(request, "username", "anonymous") or "anonymous"
    return f"{username}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


# ═════════════════════════════════════════════════
#  智能对话
# ═════════════════════════════════════════════════

def agent_chat(message, history, request: gr.Request):
    ok, err = _check_llm()
    if not ok:
        yield err
        return
    if not message.strip():
        yield "请输入你的问题"
        return
    if not _check_rate_limit("webui_chat"):
        yield "⚠️ 请求过于频繁，请稍后再试"
        return

    username = getattr(request, "username", "anonymous") or "anonymous"
    session_id = f"chat_{username}_{uuid.uuid4().hex[:6]}"

    try:
        logger.info(f"Chat session={session_id[:20]}: {message[:60]}")
        yield "🤔 正在分析你的问题..."
        response = orch.smart_chat(message, session_id=session_id)
        yield response
    except Exception as e:
        logger.error(f"chat 异常: {e}", exc_info=True)
        yield f"❌ 处理出错: {e}"


# ═════════════════════════════════════════════════
#  选品 / 上架 / 货源 / 客服 / 运营分析
# ═════════════════════════════════════════════════

def run_selector(category, budget, audience, request: gr.Request):
    ok, err = _check_llm()
    if not ok:
        return err, None
    if not category.strip():
        return "请输入品类方向", None
    username = getattr(request, "username", "anonymous") or "anonymous"
    session_id = f"sel_{username}_{uuid.uuid4().hex[:6]}"
    try:
        result = orch.run_agent("selector", category, session_id=session_id, budget=budget, target_audience=audience)
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
        return f"❌ 处理出错: {e}", None


def run_lister(name, category, features, price, cost, request: gr.Request):
    ok, err = _check_llm()
    if not ok:
        return err
    if not name.strip():
        return "请输入商品名称"
    username = getattr(request, "username", "anonymous") or "anonymous"
    session_id = f"list_{username}_{uuid.uuid4().hex[:6]}"
    try:
        info = {"name": name, "category": category, "features": features, "target_price": price, "cost": cost}
        result = orch.run_agent("lister", info, session_id=session_id)
        content = result.get("listing_content", "生成失败")
        saved = result.get("saved_to", "")
        if saved:
            content += f"\n\n---\n📁 已保存到: `{saved}`"
        return content
    except Exception as e:
        return f"❌ 处理出错: {e}"


def run_sourcing(name, category, target_price, expected_sales, budget, request: gr.Request):
    ok, err = _check_llm()
    if not ok:
        return err
    if not name.strip() and not category.strip():
        return "请输入商品名称或品类"
    username = getattr(request, "username", "anonymous") or "anonymous"
    session_id = f"src_{username}_{uuid.uuid4().hex[:6]}"
    try:
        info = {"name": name, "category": category, "target_price": target_price, "expected_sales": expected_sales, "budget": budget}
        result = orch.run_agent("sourcing", info, session_id=session_id)
        report = result.get("report", "生成失败")
        tc = result.get("tool_calls", 0)
        return report + f"\n\n---\n🤖 AI 自主调用了 {tc} 次工具分析"
    except Exception as e:
        return f"❌ 处理出错: {e}"


def run_service(query, product_context, request: gr.Request):
    ok, err = _check_llm()
    if not ok:
        return err
    if not query.strip():
        return "请输入顾客问题"
    username = getattr(request, "username", "anonymous") or "anonymous"
    session_id = f"svc_{username}_{uuid.uuid4().hex[:6]}"
    try:
        result = orch.run_agent("service", query, session_id=session_id, product_context=product_context)
        return result.get("answer", "生成失败")
    except Exception as e:
        return f"❌ 处理出错: {e}"


def run_analyst(data, cost, price, volume, request: gr.Request):
    ok, err = _check_llm()
    if not ok:
        return err, None
    if not data.strip():
        return "请输入经营数据", None
    username = getattr(request, "username", "anonymous") or "anonymous"
    session_id = f"an_{username}_{uuid.uuid4().hex[:6]}"
    try:
        pricing_data = {}
        if cost: pricing_data["cost"] = cost
        if price: pricing_data["price"] = price
        if volume: pricing_data["volume"] = volume
        result = orch.run_agent("analyst", data if not pricing_data else {**pricing_data, "data": data}, session_id=session_id)
        report = result.get("report", "生成失败")
        pricing = result.get("pricing_advice")
        pt = ""
        if pricing:
            pa = pricing.get("profit_analysis", {})
            pt = f"利润: ¥{pa.get('profit', 0)} | 利润率: {pa.get('margin', 0)}%"
            ps = pricing.get("price_suggestion", {})
            pt += f"\n建议售价: ¥{ps.get('competitive_price', 0)} (常规) / ¥{ps.get('premium_price', 0)} (溢价)"
        return report, pt if pt else None
    except Exception as e:
        return f"❌ 处理出错: {e}", None


def run_workflow(category, budget, audience, request: gr.Request):
    ok, err = _check_llm()
    if not ok:
        yield err
        return
    if not category.strip():
        yield "请输入品类方向"
        return
    username = getattr(request, "username", "anonymous") or "anonymous"
    session_id = f"wf_{username}_{uuid.uuid4().hex[:8]}"

    try:
        output = f"# 🛒 完整工作流报告：{category}\n\n"
        results = {"category": category, "session_id": session_id}

        # 步骤 1/4: 选品分析
        yield output + "⏳ **步骤 1/4：选品分析中...**\n"
        try:
            sel_result = orch.run_agent("selector", category, session_id=session_id, budget=budget, target_audience=audience)
            results["selector"] = sel_result
            output += f"## 📋 选品分析\n{sel_result.get('report', 'N/A')}\n\n"
        except Exception as e:
            output += f"## 📋 选品分析\n❌ 选品失败: {e}\n\n"
        yield output

        # 步骤 2/4: 上架素材
        yield output + "⏳ **步骤 2/4：上架素材生成中...**\n"
        try:
            list_input = f"品类: {category}\n预算: {budget}\n目标人群: {audience}"
            list_result = orch.run_agent("lister", list_input, session_id=session_id)
            results["lister"] = list_result
            output += f"## 📝 上架素材\n{list_result.get('listing_content', list_result.get('report', 'N/A'))}\n\n"
        except Exception as e:
            output += f"## 📝 上架素材\n❌ 上架失败: {e}\n\n"
        yield output

        # 步骤 3/4: 客服
        yield output + "⏳ **步骤 3/4：客服话术生成中...**\n"
        try:
            svc_result = orch.run_agent("service", f"品类: {category} 的常见客服问题", session_id=session_id)
            results["service"] = svc_result
            output += f"## 💬 客服应答\n{svc_result.get('answer', svc_result.get('report', 'N/A'))}\n\n"
        except Exception as e:
            output += f"## 💬 客服应答\n❌ 客服生成失败: {e}\n\n"
        yield output

        # 步骤 4/4: 运营分析
        yield output + "⏳ **步骤 4/4：运营分析中...**\n"
        try:
            an_result = orch.run_agent("analyst", f"新店铺启动\n品类: {category}\n预算: {budget}\n目标人群: {audience}", session_id=session_id)
            results["analyst"] = an_result
            output += f"## 📊 运营建议\n{an_result.get('report', 'N/A')}\n\n"
        except Exception as e:
            output += f"## 📊 运营建议\n❌ 分析失败: {e}\n\n"
        yield output

        output += "---\n✅ **全部完成！**"
        yield output
    except Exception as e:
        yield f"❌ 工作流出错: {e}"


# ═════════════════════════════════════════════════
#  批量上架
# ═════════════════════════════════════════════════

def parse_csv(file):
    if file is None:
        return None, "请上传 CSV 文件"
    try:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        reader = csv.DictReader(io.StringIO(content))
        products = []
        for row in reader:
            name = row.get("名称", "") or row.get("name", "") or ""
            cat = row.get("品类", "") or row.get("category", "") or ""
            feat = row.get("卖点", "") or row.get("features", "") or ""
            price = row.get("售价", "") or row.get("price", "") or ""
            cost = row.get("成本", "") or row.get("cost", "") or ""
            if name:
                products.append({"name": name, "category": cat, "features": feat, "target_price": price, "cost": cost})
        if not products:
            return None, "CSV 未找到有效数据。需要列：名称,品类,卖点,售价,成本"
        return products, f"成功解析 {len(products)} 个商品"
    except Exception as e:
        return None, f"解析失败: {e}"


def run_batch_listing(file):
    ok, err = _check_llm()
    if not ok:
        yield f"❌ {err}", None
        return
    products, msg = parse_csv(file)
    if not products:
        yield msg, None
        return

    output = ""
    all_results = []
    total = len(products)
    yield f"📦 共解析 {total} 个商品\n\n", None

    for i, product in enumerate(products, 1):
        name = product.get("name", f"商品{i}")
        yield f"⏳ ({i}/{total}) {name} 生成中...\n", all_results
        try:
            result = orch.run_agent("lister", product)
            item = {"index": i, "name": name, "content": result.get("listing_content", "生成失败")}
            all_results.append(item)
            output += f"---\n### ✅ {i}. {name}\n{item['content']}\n\n"
            yield output, all_results
        except Exception as e:
            all_results.append({"index": i, "name": name, "content": f"❌ 失败: {e}"})
            output += f"---\n### ❌ {i}. {name}\n失败: {e}\n\n"
            yield output, all_results

    output += "---\n✅ 全部完成"
    yield output, all_results


def export_batch_csv(results):
    if not results:
        return None
    import tempfile
    out = io.StringIO()
    out.write("商品名称,标题方案1,标题方案2,标题方案3,核心卖点\n")
    for r in results:
        name = r.get("name", "")
        lines = r.get("content", "").split("\n")
        # 匹配 "方案N：" 开头的标题行（如 "方案1：淘宝搜索优化版"），排除正文中随意出现的"方案"
        titles = [l for l in lines if re.match(r'^方案\d+[：:]', l.strip())]
        t1 = titles[0].split("：")[-1].strip() if len(titles) > 0 else ""
        t2 = titles[1].split("：")[-1].strip() if len(titles) > 1 else ""
        t3 = titles[2].split("：")[-1].strip() if len(titles) > 2 else ""
        out.write(f"{name},{t1},{t2},{t3},\n")
    tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".csv", delete=False)
    tmp.write(out.getvalue())
    tmp.close()
    return tmp.name


# ═════════════════════════════════════════════════
#  知识库管理
# ═════════════════════════════════════════════════

def refresh_faq_list():
    faqs = get_all_faqs()
    if not faqs:
        return "（暂无 FAQ，点击「新增 FAQ」添加）"
    lines = []
    for i, faq in enumerate(faqs):
        lines.append(f"**{i+1}. {faq['q']}**")
        lines.append(f"   {faq['a'][:120]}{'...' if len(faq['a']) > 120 else ''}")
        lines.append("")
    return "\n".join(lines)


def add_faq_entry(q, a):
    if not q.strip() or not a.strip():
        return "请输入问题和答案", refresh_faq_list()
    add_faq(q.strip(), a.strip())
    return "✅ 已添加", refresh_faq_list()


def delete_faq_entry(index_str):
    try:
        idx = int(index_str) - 1
        if delete_faq(idx):
            return f"✅ 已删除第 {index_str} 条", refresh_faq_list()
        return "❌ 无效编号", refresh_faq_list()
    except (ValueError, IndexError):
        return "❌ 请输入有效编号", refresh_faq_list()


def refresh_price_view():
    lib = get_price_library()
    if not lib:
        return "（暂无价格数据）"
    lines = []
    for cat in lib:
        lines.append(f"### {cat['category']}")
        for i, sub in enumerate(cat.get("subcategories", [])):
            lines.append(f"- {sub['name']}: ¥{sub['price_range']} | 成本¥{sub['cost_range']} | 竞争{sub.get('competition','?')}")
        lines.append("")
    return "\n".join(lines)


def add_price_item(category, name, price_range, cost_range, competition):
    if not category.strip() or not name.strip():
        return "请输入品类和子类名称", refresh_price_view()
    data = {"name": name.strip(), "price_range": price_range or "?", "cost_range": cost_range or "?", "target": "通用", "features": [], "competition": competition or "中"}
    ok = add_price_subcategory(category.strip(), data)
    return ("✅ 已添加" if ok else "❌ 品类不存在，请先确认品类名称"), refresh_price_view()


def refresh_supplier_view():
    suppliers = get_suppliers()
    if not suppliers:
        return "（暂无供应商数据）"
    lines = []
    for sup in suppliers:
        lines.append(f"### {sup['category']}")
        lines.append(f"产区: {', '.join(sup.get('sourcing_region', []))}")
        for i, p in enumerate(sup.get("products", [])):
            lines.append(f"  {i+1}. {p['name']} | 批发¥{p['wholesale_price']} | MOQ{p.get('moq','?')}")
        lines.append("")
    return "\n".join(lines)


def add_supplier_item(category, name, wholesale_price, moq, region):
    if not category.strip() or not name.strip():
        return "请输入品类和产品名称", refresh_supplier_view()
    data = {"name": name.strip(), "wholesale_price": wholesale_price or "?", "moq": moq or "?", "moq_price": "?", "supplier_types": [region or "未知"], "sourcing_notes": ""}
    ok = add_supplier(category.strip(), data)
    return ("✅ 已添加" if ok else "❌ 添加失败"), refresh_supplier_view()


def refresh_template_view():
    templates = get_all_templates()
    if not templates:
        return "（暂无模板）"
    lines = []
    for t in templates:
        lines.append(f"### {t['category']}")
        for tp in t.get("title_patterns", [])[:2]:
            lines.append(f"- 标题: {tp}")
        lines.append(f"- SEO: {', '.join(t.get('seo_keywords', [])[:3])}")
        lines.append("")
    return "\n".join(lines)


# ═════════════════════════════════════════════════
#  系统监控
# ═════════════════════════════════════════════════

def refresh_system_status():
    try:
        stats = get_stats(24)
        daily = get_daily_stats(7)
        recent = get_recent_calls(10)
    except Exception:
        return "（监控数据库尚未就绪，使用后自动生成数据）", ""

    lines = [
        f"### 概览（最近 24 小时）",
        f"- 总调用次数: {stats['calls']}",
        f"- 成功: {stats['success']} | 失败: {stats['failed']}",
        f"- 总费用: ¥{stats['total_cost']}",
        f"- 平均响应: {stats['avg_duration_ms']}ms",
        f"",
        f"### 按 Provider",
    ]
    for prov, data in stats["by_provider"].items():
        lines.append(f"- {prov}: {data['calls']} 次, ¥{data['cost']}")
    lines.append("")
    lines.append(f"### 每日趋势")
    for d in daily:
        lines.append(f"- {d['day']}: {d['calls']} 次, ¥{d['cost']}, {d['avg_duration_ms']}ms")

    recent_lines = ["### 最近调用"]
    for r in recent[-10:]:
        status = "✅" if r.get("success") else "❌"
        recent_lines.append(f"- {status} {r.get('provider','?')}/{r.get('model','?')} {r.get('duration_ms',0)}ms | {r.get('agent','')}")
    return "\n".join(lines), "\n".join(recent_lines)


# ═════════════════════════════════════════════════
#  构建界面
# ═════════════════════════════════════════════════

def build_app():
    with gr.Blocks(title="商家智能运营 Agent", theme="soft") as app:
        gr.Markdown("""# 🛒 商家智能运营 Agent
        👉 选品 → 上架 → 客服 → 运营分析 — AI 全流程自动完成
        """)

        with gr.Tabs():
            # ===== Tab 1: 智能对话 =====
            with gr.TabItem("💬 智能对话"):
                gr.Markdown("随便问，自动路由到对应 Agent。**支持多轮上下文记忆。**")
                gr.ChatInterface(
                    agent_chat,
                    title="智能电商助手",
                    description="输入任何电商相关问题",
                    additional_inputs=[],
                )

            # ===== Tab 2: 选品 =====
            with gr.TabItem("📋 选品分析"):
                with gr.Row():
                    with gr.Column():
                        category_in = gr.Textbox(label="品类方向", placeholder="如: 学生文具、宠物用品")
                        budget_in = gr.Textbox(label="启动预算（可选）", placeholder="如: 10000")
                        audience_in = gr.Textbox(label="目标人群（可选）", placeholder="如: 大学生")
                        btn_selector = gr.Button("开始分析", variant="primary")
                    with gr.Column():
                        selector_out = gr.Markdown(label="分析报告")
                        recs_out = gr.Markdown(label="推荐商品评分")
                btn_selector.click(run_selector, [category_in, budget_in, audience_in], [selector_out, recs_out])

            # ===== Tab 3: 上架 =====
            with gr.TabItem("📝 上架素材"):
                with gr.Row():
                    with gr.Column():
                        name_in = gr.Textbox(label="商品名称", placeholder="如: 智能保温杯")
                        cat_in = gr.Textbox(label="所属品类", placeholder="如: 家居日用")
                        feat_in = gr.Textbox(label="核心卖点（逗号分隔）", placeholder="如: 24小时保温,316不锈钢")
                        price_in = gr.Textbox(label="目标售价", placeholder="如: 89")
                        cost_in = gr.Textbox(label="成本价", placeholder="如: 35")
                        btn_lister = gr.Button("生成素材", variant="primary")
                    with gr.Column():
                        lister_out = gr.Markdown(label="上架素材")
                btn_lister.click(run_lister, [name_in, cat_in, feat_in, price_in, cost_in], lister_out)

            # ===== Tab 4: 批量上架 =====
            with gr.TabItem("📦 批量上架"):
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="上传 CSV", file_types=[".csv"])
                        btn_batch = gr.Button("🚀 批量生成", variant="primary")
                    with gr.Column():
                        batch_out = gr.Markdown(label="结果")
                state = gr.State()
                download_file = gr.File(label="下载 CSV")
                btn_batch.click(run_batch_listing, file_input, [batch_out, state])
                state.change(export_batch_csv, state, download_file)

            # ===== Tab 5: 货源 =====
            with gr.TabItem("🏭 一手货源"):
                with gr.Row():
                    with gr.Column():
                        src_name = gr.Textbox(label="商品名称", placeholder="如: 迷你手持小风扇")
                        src_cat = gr.Textbox(label="所属品类", placeholder="如: 电风扇")
                        src_price = gr.Textbox(label="目标售价（可选）", placeholder="如: 39")
                        src_sales = gr.Textbox(label="预期月销量（可选）", placeholder="如: 500")
                        src_budget = gr.Textbox(label="采购预算（可选）", placeholder="如: 5000")
                        btn_sourcing = gr.Button("分析一手货源", variant="primary")
                    with gr.Column():
                        sourcing_out = gr.Markdown(label="货源分析报告")
                btn_sourcing.click(run_sourcing, [src_name, src_cat, src_price, src_sales, src_budget], sourcing_out)

            # ===== Tab 6: 客服 =====
            with gr.TabItem("💬 客服应答"):
                with gr.Row():
                    with gr.Column():
                        query_in = gr.Textbox(label="顾客问题", placeholder="如: 什么时候发货？", lines=3)
                        ctx_in = gr.Textbox(label="商品信息（可选）", placeholder="填写商品名称/规格/价格", lines=2)
                        btn_service = gr.Button("生成回复", variant="primary")
                    with gr.Column():
                        service_out = gr.Markdown(label="客服回复")
                btn_service.click(run_service, [query_in, ctx_in], service_out)

            # ===== Tab 7: 运营分析 =====
            with gr.TabItem("📊 运营分析"):
                with gr.Row():
                    with gr.Column():
                        data_in = gr.Textbox(label="经营数据", placeholder="店铺A上周销售数据：商品卖出200件...", lines=5)
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
            with gr.TabItem("🚀 一键工作流"):
                with gr.Row():
                    with gr.Column():
                        wf_category = gr.Textbox(label="品类方向", placeholder="如: 宠物用品")
                        wf_budget = gr.Textbox(label="预算（可选）", placeholder="如: 5000")
                        wf_audience = gr.Textbox(label="目标人群（可选）", placeholder="如: 养猫人群")
                        btn_workflow = gr.Button("🚀 一键执行", variant="primary")
                    with gr.Column():
                        wf_out = gr.Markdown(label="工作流报告")
                btn_workflow.click(run_workflow, [wf_category, wf_budget, wf_audience], wf_out)

            # ===== Tab 9: 知识库管理 =====
            with gr.TabItem("📚 知识库管理"):
                with gr.Tabs():
                    # FAQ 管理
                    with gr.TabItem("FAQ 问答"):
                        faq_display = gr.Markdown(refresh_faq_list())
                        with gr.Row():
                            faq_q = gr.Textbox(label="问题", placeholder="如: 什么时候发货", scale=2)
                            faq_a = gr.Textbox(label="答案", placeholder="如: 工作日24小时内发货...", scale=3)
                        with gr.Row():
                            btn_faq_add = gr.Button("新增 FAQ", variant="primary")
                            faq_del_idx = gr.Textbox(label="删除编号", placeholder="输入要删除的 FAQ 编号", scale=1)
                            btn_faq_del = gr.Button("删除")
                        btn_faq_add.click(add_faq_entry, [faq_q, faq_a], [faq_q, faq_display])
                        btn_faq_del.click(delete_faq_entry, [faq_del_idx], [faq_del_idx, faq_display])

                    # 价格库管理
                    with gr.TabItem("价格库"):
                        price_display = gr.Markdown(refresh_price_view)
                        with gr.Row():
                            pc_cat = gr.Textbox(label="品类", placeholder="如: 宠物用品")
                            pc_name = gr.Textbox(label="子类名称", placeholder="如: 猫抓板")
                        with gr.Row():
                            pc_price = gr.Textbox(label="价格范围", placeholder="如: 15-50")
                            pc_cost = gr.Textbox(label="成本范围", placeholder="如: 5-15")
                            pc_comp = gr.Dropdown(label="竞争度", choices=["极低", "低", "中低", "中", "高", "极高"], value="中")
                        btn_price_add = gr.Button("新增价格条目", variant="primary")
                        btn_price_add.click(add_price_item, [pc_cat, pc_name, pc_price, pc_cost, pc_comp], [pc_cat, price_display])

                    # 供应商管理
                    with gr.TabItem("供应商库"):
                        sup_display = gr.Markdown(refresh_supplier_view)
                        with gr.Row():
                            sc_cat = gr.Textbox(label="品类", placeholder="如: 宠物用品")
                            sc_name = gr.Textbox(label="产品名称", placeholder="如: 猫抓板")
                        with gr.Row():
                            sc_wp = gr.Textbox(label="批发价范围", placeholder="如: 3-12")
                            sc_moq = gr.Textbox(label="MOQ", placeholder="如: 50-300")
                            sc_region = gr.Textbox(label="产区", placeholder="如: 河北邢台")
                        btn_sup_add = gr.Button("新增供应商产品", variant="primary")
                        btn_sup_add.click(add_supplier_item, [sc_cat, sc_name, sc_wp, sc_moq, sc_region], [sc_cat, sup_display])

                    # 上架模板
                    with gr.TabItem("上架模板"):
                        gr.Markdown("当前已有的品类模板：")
                        tmpl_display = gr.Markdown(refresh_template_view)
                        gr.Markdown("> 模板仅支持通过 JSON 文件直接编辑：`knowledge/data/product_templates.json`")

            # ===== Tab 10: 系统状态 =====
            with gr.TabItem("📊 系统状态"):
                gr.Markdown("### LLM 调用监控（最近 24 小时）")
                refresh_btn = gr.Button("🔄 刷新数据")
                status_out = gr.Markdown("点击刷新加载数据...")
                recent_out = gr.Markdown()
                refresh_btn.click(refresh_system_status, outputs=[status_out, recent_out])
                # 页面加载时自动刷新
                app.load(refresh_system_status, outputs=[status_out, recent_out])

        gr.Markdown(f"---\n💡 首次使用请用默认账号 `{ADMIN_USER} / {ADMIN_PASS}` 登录")

    return app


def main():
    # 启动时自动检查
    ok, _, _ = check_env()
    if not ok:
        logger.warning(f"LLM({LLM_PROVIDER}) 不可用，启动后功能受限")

    app = build_app()
    app.queue()
    app.launch(
        server_name="127.0.0.1",
        server_port=WEB_PORT,
        auth=check_login,
        auth_message=f"请输入用户名和密码（默认: {ADMIN_USER} / {ADMIN_PASS}）",
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
