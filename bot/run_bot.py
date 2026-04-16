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
        text="Katalogga o'tish", 
        web_app=WebAppInfo(url=WEBAPP_URL)
    ))

    await message.answer(
        f"Assalomu alaykum, {message.from_user.full_name}!\n\n"
        "TanlaAI - eshiklarni intellektual tanlash va vizualizatsiya qilish platformasiga xush kelibsiz.",
        reply_markup=builder.as_markup()
    )


# --- DJANGO SETUP FOR BOT ---
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
import django
django.setup()

from asgiref.sync import sync_to_async
from shop.models import LeadRequest

@sync_to_async
def update_lead_status(lead_id: str, new_status: str):
    obj = LeadRequest.objects.filter(id=lead_id).first()
    if obj:
        obj.status = new_status
        obj.save()
        return True
    return False

@dp.callback_query(lambda c: c.data and (c.data.startswith('sold_') or c.data.startswith('cancel_')))
async def process_lead_status(callback_query: types.CallbackQuery):
    action, lead_id = callback_query.data.split('_', 1)
    
    if action == 'sold':
        success = await update_lead_status(lead_id, 'converted')
        text = "✅ Sotildi! (Konversiya qayd etildi)"
    else:
        success = await update_lead_status(lead_id, 'rejected')
        text = "❌ Bekor qilindi."

    if success:
        await callback_query.answer(text, show_alert=True)
        # Edit the inline keyboard to remove the choice buttons but keep the phone number
        # Just grab the existing keyboard and build a new one without the action row
        message_text = callback_query.message.text
        message_text += f"\n\n<b>Status:</b> {text}"
        
        # remove inline keyboard entirely to avoid double clicking
        await callback_query.message.edit_text(message_text, parse_mode="HTML", reply_markup=None)
    else:
        await callback_query.answer("Topilmadi yoki allaqachon o'zgartirilgan", show_alert=True)

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
