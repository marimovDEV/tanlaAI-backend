import requests
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class NotificationService:
    @staticmethod
    def send_telegram_message(message: str, chat_id: str = None, reply_markup: dict = None):
        """
        Sends a message to a Telegram chat. 
        Defaults to settings.ADMIN_TELEGRAM_ID if chat_id is not provided.
        """
        token = getattr(settings, 'TELEGRAM_BOT_TOKEN', None)
        # Use a specific admin ID from settings OR fallback to a generic one
        target_chat_id = chat_id or getattr(settings, 'ADMIN_TELEGRAM_ID', None)

        if not token or not target_chat_id:
            logger.warning("Telegram notification skipped: Token or Chat ID missing.")
            return False

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": target_chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False

    @staticmethod
    def notify_new_lead(lead):
        """
        Formats and sends a notification about a new lead.
        """
        product_name = lead.product.name
        user_name = f"{lead.user.first_name} {lead.user.last_name or ''}".strip()
        phone = lead.phone or "Ko'rsatilmagan"
        lead_type = lead.get_lead_type_display()
        
        message = (
            f"🎯 <b>Yangi Lead!</b>\n\n"
            f"👤 <b>Mijoz:</b> {user_name}\n"
            f"📞 <b>Telefon:</b> {phone}\n"
            f"📦 <b>Mahsulot:</b> {product_name}\n"
            f"🛠 <b>Tur:</b> {lead_type}\n"
        )
        
        if lead.price_info:
            message += f"💰 <b>O'lcham/Narx:</b> {lead.price_info}\n"
            
        if lead.message:
            message += f"\n📝 <b>Izoh:</b> {lead.message}"
            
        message += f"\n\n🚀 <a href='https://tanla-ai.ardentsoft.uz/adminka/leads'>Admin panelda ko'rish</a>"

        # Build inline keyboard if phone exists
        reply_markup = {"inline_keyboard": []}

        if lead.phone:
            reply_markup["inline_keyboard"].append([{"text": "📞 Uyali aloqa / Bog'lanish", "url": f"tel:{lead.phone}"}])

        reply_markup["inline_keyboard"].append([
            {"text": "✅ Sotildi", "callback_data": f"sold_{lead.id}"},
            {"text": "❌ Bekor", "callback_data": f"cancel_{lead.id}"}
        ])

        # Notify Admin
        NotificationService.send_telegram_message(message, reply_markup=reply_markup)
        
        # Notify Company Owner if applicable
        if lead.company and lead.company.user and lead.company.user.telegram_id:
             message_company = message + "\n\n<i>Sizning kompaniyangizga yangi so'rov tushdi!</i>"
             NotificationService.send_telegram_message(message_company, chat_id=str(lead.company.user.telegram_id), reply_markup=reply_markup)
