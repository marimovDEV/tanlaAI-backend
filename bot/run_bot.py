import asyncio
import logging
import os
import sys
import socket
import aiohttp
from typing import Any, Optional
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.types import WebAppInfo, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.session.aiohttp import AiohttpSession, _prepare_connector

import environ

# Load environment variables
env = environ.Env()
environ.Env.read_env(os.path.join(os.path.dirname(__file__), '..', '.env'))

TOKEN = env('TELEGRAM_BOT_TOKEN')
WEBAPP_URL = env('NGROK_URL')
PROXY_URL = env('PROXY_URL', default=None)

# Custom Session to force IPv4 in aiogram 3.x with optional Proxy support
class IPv4Session(AiohttpSession):
    async def create_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector_type = aiohttp.TCPConnector
            connector_init = {"family": socket.AF_INET, "enable_cleanup_closed": True}

            if self.proxy:
                # Use aiohttp_socks to handle proxy with fixed family
                connector_type, connector_init = _prepare_connector(self.proxy)
                # Ensure the proxy connector also uses IPv4 if possible
                # (most proxy connectors handle this through the proxy server)

            self._session = aiohttp.ClientSession(
                connector=connector_type(**connector_init),
                json_serialize=self.json_dumps,
            )
        return self._session

dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: types.Message) -> None:
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(
        text="Open ATELIER", 
        web_app=WebAppInfo(url=WEBAPP_URL)
    ))

    await message.answer(
        f"Welcome to ATELIER, {message.from_user.full_name}!\n\n"
        "Experience our digital boutique directly in Telegram.",
        reply_markup=builder.as_markup()
    )

async def main() -> None:
    # Use our custom IPv4 session with optional proxy
    session = IPv4Session(proxy=PROXY_URL)
    bot = Bot(token=TOKEN, session=session)
    
    try:
        # Check bot status
        bot_user = await bot.get_me()
        print(f"DEBUG: [Bot Service] Started as @{bot_user.username}")
        await dp.start_polling(bot)
    finally:
        await session.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
