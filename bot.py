import os
import asyncio
import tempfile
import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

import requests
from pathlib import Path
from typing import List
from pdfminer.high_level import extract_text
from openai import OpenAI

# --------- تنظیمات و لاگ ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pdf-summarizer-bot")

# --------- متغیرهای محیطی ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is missing! Bot will not work.")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is missing. Summaries will fail.")

# ساخت کلاینت OpenAI
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --------- کمک‌تابع‌ها ----------
def chunk_text(text: str, max_chars: int = 10_000) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

async def summarize_text(text: str) -> str:
    if not client:
        return "خطا: کلید OPENAI_API_KEY تنظیم نشده است."
    chunks = chunk_text(text, max_chars=10_000)
    partial_summaries = []
    for i, ch in enumerate(chunks, start=1):
        logger.info(f"Summarizing chunk {i}/{len(chunks)} (size={len(ch)})")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "تو یک خلاصه‌ساز دقیق و منظم هستی."},
                {"role": "user", "content": f"این بخش از متن را خلاصه کن:\n\n{ch}"},
            ],
            temperature=0.3,
        )
        partial_summaries.append(resp.choices[0].message.content.strip())

    final_input = "\n\n---\n\n".join(partial_summaries)
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "خلاصه‌ی نهایی و منسجم با تیترهای کوتاه تولید کن."},
            {"role": "user", "content": f"این خلاصه‌های بخش‌بندی‌شده را به یک خلاصه‌ی نهایی تبدیل کن:\n\n{final_input}"},
        ],
        temperature=0.3,
    )
    return final.choices[0].message.content.strip()

def download_file(file_url: str, dest_path: Path) -> None:
    with requests.get(file_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# --------- هندلرها ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! فایل PDF خودت رو بفرست تا برات خلاصه کنم.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("راهنما:\n- دستور /start\n- یک فایل PDF بفرست\n")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    document = message.document
    if not document or document.mime_type != "application/pdf":
        await message.reply_text("لطفاً فقط فایل PDF ارسال کن.")
        return
    try:
        file = await context.bot.get_file(document.file_id)
        file_url = file.file_path
        await message.reply_text("فایل دریافت شد. در حال پردازش...")
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / (document.file_name or "file.pdf")
            download_file(file_url, pdf_path)
            text = extract_text(str(pdf_path))
            if not text.strip():
                await message.reply_text("متن قابل استخراج پیدا نشد.")
                return
            summary = await summarize_text(text)
            await message.reply_text(summary)
    except Exception as e:
        logger.exception("Error while processing document:")
        await message.reply_text(f"خطا: {e}")

async def unknown_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("دستور ناشناخته.")

# --------- بوت اپلیکیشن ----------
def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("Bot cannot start without TELEGRAM_BOT_TOKEN.")
        return
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_cmd))
    logger.info("Bot is starting polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
