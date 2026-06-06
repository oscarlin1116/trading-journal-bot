import os, hashlib, hmac, base64, json, re, sqlite3, httpx, contextlib
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import anthropic

load_dotenv()

LINE_SECRET = os.environ["LINE_CHANNEL_SECRET"]
LINE_TOKEN  = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
ANT_KEY     = os.environ["ANTHROPIC_API_KEY"]
DB_PATH     = os.environ.get("DB_PATH", "trades.db")

app = FastAPI(title="交易紀錄 Bot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Database ──────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT,
                code        TEXT,
                trade_type  TEXT,
                shares      INTEGER,
                date_buy    TEXT,
                date_sell   TEXT,
                price_buy   REAL,
                price_sell  REAL,
                amount_buy  REAL DEFAULT 0,
                amount_sell REAL DEFAULT 0,
                pnl         REAL,
                return_rate REAL,
                fee         REAL DEFAULT 0,
                tax         REAL DEFAULT 0,
                source      TEXT DEFAULT 'line',
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

init_db()

# ── LINE helpers ──────────────────────────────────────────────────────────────

def verify_sig(body: bytes, sig: str) -> bool:
    mac = hmac.new(LINE_SECRET.encode(), body, hashlib.sha256).digest()
    return hmac.compare_digest(base64.b64encode(mac).decode(), sig)

async def download_image(msg_id: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"https://api-data.line.me/v2/bot/message/{msg_id}/content",
            headers={"Authorization": f"Bearer {LINE_TOKEN}"},
        )
        r.raise_for_status()
        media_type = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        return r.content, media_type

async def line_reply(reply_token: str, text: str):
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"},
            json={"replyToken": reply_token, "messages": [{"type": "text", "text": text}]},
        )

# ── Claude Vision ─────────────────────────────────────────────────────────────

PROMPT = """
你是台股券商截圖辨識助理。請從這張「已實現損益」截圖辨識所有交易，回傳 JSON 陣列。

每筆格式：
{
  "symbol": "股票名稱",
  "code": "代號（如 2330）",
  "trade_type": "現股 或 融資 或 融券",
  "shares": 股數整數,
  "date_buy": "YYYY-MM-DD",
  "date_sell": "YYYY-MM-DD",
  "price_buy": 買進均價數字,
  "price_sell": 賣出均價數字,
  "amount_buy": 買進金額（沒有填0）,
  "amount_sell": 賣出金額（沒有填0）,
  "pnl": 損益（虧損為負）,
  "return_rate": 報酬率數字（7.5 代表 7.5%）,
  "fee": 手續費（沒有填0）,
  "tax": 交易稅（沒有填0）
}

只回傳 JSON 陣列，不要其他文字。
若不是已實現損益截圖，回傳：{"error": "請傳已實現損益的截圖"}
"""

async def analyze_image(image_bytes: bytes, media_type: str) -> list[dict]:
    client = anthropic.Anthropic(api_key=ANT_KEY)
    b64 = base64.b64encode(image_bytes).decode()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    text = msg.content[0].text.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        obj = json.loads(m.group())
        if "error" in obj:
            raise ValueError(obj["error"])
    raise ValueError("無法解析辨識結果，請重新傳一次")

# ── Webhook ───────────────────────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    data = json.loads(body)

    # LINE verification ping (empty events) — skip sig check
    if not data.get("events"):
        return {"ok": True}
    for event in data.get("events", []):
        if event.get("type") != "message":
            continue
        reply_token = event["replyToken"]
        msg = event["message"]

        if msg["type"] == "image":
            try:
                image_bytes, media_type = await download_image(msg["id"])
                trades = await analyze_image(image_bytes, media_type)
                with db() as conn:
                    for t in trades:
                        conn.execute(
                            """INSERT INTO trades
                               (symbol,code,trade_type,shares,date_buy,date_sell,
                                price_buy,price_sell,amount_buy,amount_sell,
                                pnl,return_rate,fee,tax)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (t.get("symbol"), t.get("code"), t.get("trade_type"),
                             t.get("shares"), t.get("date_buy"), t.get("date_sell"),
                             t.get("price_buy"), t.get("price_sell"),
                             t.get("amount_buy", 0), t.get("amount_sell", 0),
                             t.get("pnl"), t.get("return_rate"),
                             t.get("fee", 0), t.get("tax", 0)),
                        )
                symbols = "、".join(dict.fromkeys(t["symbol"] for t in trades if t.get("symbol")))
                await line_reply(reply_token, f"✅ 辨識完成，新增 {len(trades)} 筆\n{symbols}")
            except ValueError as e:
                await line_reply(reply_token, f"⚠️ {e}")
            except Exception as e:
                await line_reply(reply_token, f"❌ 錯誤：{str(e)[:80]}")

        elif msg["type"] == "text":
            await line_reply(reply_token, "請傳券商 App 的「已實現損益」截圖，我會自動辨識並記錄。")

    return {"ok": True}

# ── REST API for frontend ─────────────────────────────────────────────────────

@app.get("/api/trades")
def api_trades():
    with db() as conn:
        rows = conn.execute("SELECT * FROM trades ORDER BY date_sell DESC, id DESC").fetchall()
    return [dict(r) for r in rows]

@app.delete("/api/trades/{trade_id}")
def api_delete_trade(trade_id: int):
    with db() as conn:
        conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}
