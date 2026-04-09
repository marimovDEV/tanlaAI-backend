import asyncio
import logging
import os
import sys
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.types import WebAppInfo, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder

import environ

# Load environment variables
env = environ.Env()
environ.Env.read_env(os.path.join(os.path.dirname(__file__), '..', '.env'))

TOKEN = env('TELEGRAM_BOT_TOKEN')
WEBAPP_URL = env('NGROK_URL')

dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: types.Message) -> None:
    """
    This handler receives messages with `/start` command
    """
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
    bot = Bot(token=TOKEN)
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
