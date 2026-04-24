import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.services.ai_response import get_ai_response

logger = logging.getLogger(__name__)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply when the user starts the bot."""
    if update.effective_message is None:
        return

    await update.effective_message.reply_text(
        "Xin chao. Bot dang nhan update tu FastAPI webhook va xu ly theo kieu async."
    )


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming text messages asynchronously."""
    if update.effective_message is None or not update.effective_message.text:
        return

    incoming_text = update.effective_message.text.strip()
    try:
        ai_response = await get_ai_response(incoming_text)
    except Exception:
        logger.exception("Failed to generate AI response.")
        ai_response = "He thong tam thoi gap loi. Ban vui long thu lai sau."

    await update.effective_message.reply_text(ai_response)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Centralized async error logging for PTB handlers."""
    logger.exception("Telegram update processing failed: %s", context.error)


def register_handlers(application: Application) -> None:
    """Attach Telegram handlers to the PTB application."""
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_error_handler(error_handler)
