import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException, Request, status
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import Application

from app.bot.handlers import register_handlers
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

settings = get_settings()


def build_telegram_application() -> Application:
    """Create a PTB application configured for custom webhook mode."""
    application = Application.builder().token(settings.token).updater(None).build()
    register_handlers(application)
    return application


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop the Telegram application alongside FastAPI."""
    telegram_app = build_telegram_application()
    app.state.telegram_app = telegram_app

    await telegram_app.initialize()
    await telegram_app.start()

    try:
        await telegram_app.bot.set_webhook(
            url=settings.telegram_webhook_url,
            allowed_updates=Update.ALL_TYPES,
        )
        logger.info("Telegram webhook registered at %s", settings.telegram_webhook_url)
    except TelegramError:
        await telegram_app.stop()
        await telegram_app.shutdown()
        raise

    try:
        yield
    finally:
        with suppress(TelegramError):
            await telegram_app.bot.delete_webhook(drop_pending_updates=False)
        await telegram_app.stop()
        await telegram_app.shutdown()
        logger.info("Telegram application stopped cleanly.")


app = FastAPI(
    title="DLU Telegram Bot Backend",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook", status_code=status.HTTP_202_ACCEPTED, tags=["telegram"])
async def telegram_webhook(request: Request) -> dict[str, bool]:
    """Receive Telegram updates and enqueue them for PTB processing."""
    payload = await request.json()
    telegram_app: Application = request.app.state.telegram_app
    update = Update.de_json(payload, telegram_app.bot)

    if update is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload is not a valid Telegram update.",
        )

    await telegram_app.update_queue.put(update)
    return {"ok": True}
