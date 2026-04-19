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
            f"📩 <b>YANGI BUYURTMA</b>\n\n"
            f"🚪 <b>Mahsulot:</b> {product_name}\n"
            f"🛠 <b>Tur:</b> {lead_type}\n"
            f"👤 <b>Mijoz:</b> {user_name}\n"
            f"📞 <b>Telefon:</b> {phone}\n"
        )

        # Structured measurement (new fields)
        if lead.width_cm and lead.height_cm:
            area_m2 = (lead.width_cm * lead.height_cm) / 10000
            message += (
                f"📐 <b>O'lcham:</b> {lead.width_cm} × {lead.height_cm} sm\n"
                f"📏 <b>Maydon:</b> {area_m2:.2f} m²\n"
            )
        if lead.calculated_price:
            try:
                price_fmt = f"{float(lead.calculated_price):,.0f}".replace(",", " ")
            except Exception:
                price_fmt = str(lead.calculated_price)
            message += f"💰 <b>Narx:</b> {price_fmt} so'm\n"

        # Legacy free-text price info (kept for compatibility — shown only if
        # we don't already have structured measurement data)
        if lead.price_info and not (lead.width_cm or lead.calculated_price):
            message += f"💰 <b>O'lcham/Narx:</b> {lead.price_info}\n"

        # Lead-time / tayyor bo'lish muddati
        lead_time = getattr(lead.product, "lead_time_days", None)
        if lead_time:
            message += f"⏱ <b>Tayyor bo'lishi:</b> {lead_time} kun\n"

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

    @staticmethod
    def notify_lead_reminder(lead):
        """
        Sends a reminder to the admin/company if the lead is still new after 10 minutes.
        """
        user_name = f"{lead.user.first_name} {lead.user.last_name or ''}".strip()
        phone = lead.phone or "Ko'rsatilmagan"
        
        message = (
            f"⏰ <b>Eslatma!</b>\n\n"
            f"👤 <b>Mijoz:</b> {user_name}\n"
            f"📞 <b>Telefon:</b> {phone}\n\n"
            f"⚠️ <b>Hali javob berilmadi!</b>"
        )
        
        # Build original inline keyboard
        reply_markup = {"inline_keyboard": []}
        if lead.phone:
            reply_markup["inline_keyboard"].append([{"text": "📞 Uyali aloqa / Bog'lanish", "url": f"tel:{lead.phone}"}])
        reply_markup["inline_keyboard"].append([
            {"text": "✅ Sotildi", "callback_data": f"sold_{lead.id}"},
            {"text": "❌ Bekor", "callback_data": f"cancel_{lead.id}"}
        ])

        # Notify Admin
        NotificationService.send_telegram_message(message, reply_markup=reply_markup)
        
        # Notify Company Owner
        if lead.company and lead.company.user and lead.company.user.telegram_id:
             NotificationService.send_telegram_message(message, chat_id=str(lead.company.user.telegram_id), reply_markup=reply_markup)
