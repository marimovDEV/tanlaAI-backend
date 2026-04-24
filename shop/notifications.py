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
    def send_telegram_location(latitude: float, longitude: float, chat_id: str = None):
        """
        Sends a native Telegram location pin (opens in built-in map).
        """
        token = getattr(settings, 'TELEGRAM_BOT_TOKEN', None)
        target_chat_id = chat_id or getattr(settings, 'ADMIN_TELEGRAM_ID', None)
        if not token or not target_chat_id:
            return False

        url = f"https://api.telegram.org/bot{token}/sendLocation"
        payload = {
            "chat_id": target_chat_id,
            "latitude": float(latitude),
            "longitude": float(longitude),
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram location: {e}")
            return False

    @staticmethod
    def upload_photo_to_telegram(photo_path: str, caption: str = "", chat_id: str = None) -> str:
        """
        Uploads a photo to Telegram and returns the file_id.
        """
        from .models import SystemSettings
        token = getattr(settings, 'TELEGRAM_BOT_TOKEN', None)
        
        target_chat_id = chat_id or getattr(SystemSettings.get_solo(), 'ai_storage_channel_id', None) or getattr(settings, 'ADMIN_TELEGRAM_ID', None)
        
        if not token or not target_chat_id:
            logger.warning("Telegram photo upload skipped: Token or Chat ID missing.")
            return None

        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        
        try:
            with open(photo_path, 'rb') as photo_file:
                payload = {
                    "chat_id": target_chat_id,
                    "caption": caption,
                    "parse_mode": "HTML"
                }
                files = {"photo": photo_file}
                response = requests.post(url, data=payload, files=files, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                if data.get("ok") and data.get("result", {}).get("photo"):
                    # Get the largest resolution photo file_id
                    photos = data["result"]["photo"]
                    return photos[-1]["file_id"]
                return None
        except Exception as e:
            logger.error(f"Failed to upload photo to Telegram: {e}")
            return None

    @staticmethod
    def send_media_group_to_telegram(photo_paths: list, caption: str = "", chat_id: str = None) -> list:
        """
        Sends multiple photos as an album (media group).
        Returns a list of file_ids for the uploaded photos.
        """
        import json
        from .models import SystemSettings
        token = getattr(settings, 'TELEGRAM_BOT_TOKEN', None)
        target_chat_id = chat_id or getattr(SystemSettings.get_solo(), 'ai_storage_channel_id', None) or getattr(settings, 'ADMIN_TELEGRAM_ID', None)

        if not token or not target_chat_id or not photo_paths:
            return []

        url = f"https://api.telegram.org/bot{token}/sendMediaGroup"
        
        media = []
        files = {}
        opened_files = []

        try:
            for i, path in enumerate(photo_paths):
                file_key = f"photo{i}"
                f = open(path, 'rb')
                opened_files.append(f)
                files[file_key] = f
                
                item = {
                    "type": "photo",
                    "media": f"attach://{file_key}",
                }
                # Attach caption to the FIRST photo in the group (standard behavior)
                if i == len(photo_paths) - 1: # Last one is usually the 'After' image
                    item["caption"] = caption
                    item["parse_mode"] = "HTML"
                
                media.append(item)

            # Note: For albums, the caption should usually be on the first OR last image.
            # We'll put it on the last one (the result) if it's Before->After order.
            # Wait, if Before is 0 and After is 1, then i=1 is the last.

            payload = {
                "chat_id": target_chat_id,
                "media": json.dumps(media)
            }
            
            response = requests.post(url, data=payload, files=files, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            file_ids = []
            if data.get("ok") and isinstance(data.get("result"), list):
                for msg in data["result"]:
                    if msg.get("photo"):
                        file_ids.append(msg["photo"][-1]["file_id"])
            return file_ids
        except Exception as e:
            logger.error(f"Failed to send media group to Telegram: {e}")
            return []
        finally:
            for f in opened_files:
                f.close()

    @staticmethod
    def notify_new_lead(lead):
        """
        Formats and sends a notification about a new lead.
        """
        product_name = lead.product.name
        user_name = f"{lead.user.first_name} {lead.user.last_name or ''}".strip()
        phone = lead.phone or "Ko'rsatilmagan"
        lead_type = lead.get_lead_type_display()

        # Header changes per lead type so the recipient can prioritize at a
        # glance — direct orders are the highest-intent (no AI detour), so
        # they get the most attention-grabbing header.
        if lead.lead_type == "direct":
            header = "🛒 <b>TO'G'RIDAN-TO'G'RI BUYURTMA</b>"
        elif lead.lead_type == "visualize":
            header = "🎨 <b>AI VIZUALIZATSIYA BUYURTMASI</b>"
        elif lead.lead_type == "measurement":
            header = "📐 <b>O'LCHASH BUYURTMASI</b>"
        elif lead.lead_type == "call":
            header = "📞 <b>QO'NG'IROQ SO'ROVI</b>"
        else:
            header = "📩 <b>YANGI BUYURTMA</b>"

        message = f"{header}\n\n"
        
        # Section: Product Details
        message += "📦 <b>BUYURTMA TAFSILOTLARI</b>\n"
        message += f"🚪 <b>Mahsulot:</b> {product_name}\n"
        
        if hasattr(lead, 'quantity') and lead.quantity > 1:
            message += f"🔢 <b>Soni:</b> {lead.quantity} ta\n"
            
        if hasattr(lead, 'total_price') and lead.total_price:
            try:
                total_fmt = f"{float(lead.total_price):,.0f}".replace(",", " ")
                message += f"💰 <b>Jami summa:</b> {total_fmt} so'm\n"
            except Exception:
                message += f"💰 <b>Jami summa:</b> {lead.total_price} so'm\n"
        elif lead.calculated_price:
            try:
                price_fmt = f"{float(lead.calculated_price):,.0f}".replace(",", " ")
                message += f"💰 <b>Narx:</b> {price_fmt} so'm\n"
            except Exception:
                message += f"💰 <b>Narx:</b> {lead.calculated_price} so'm\n"

        # Section: Measurement (if applicable)
        if lead.width_cm and lead.height_cm:
            area_m2 = (lead.width_cm * lead.height_cm) / 10000
            message += (
                f"📐 <b>O'lcham:</b> {lead.width_cm} × {lead.height_cm} sm\n"
                f"📏 <b>Maydon:</b> {area_m2:.2f} m²\n"
            )

        # Section: Customer Details
        message += "\n👤 <b>MIJOZ MA'LUMOTLARI</b>\n"
        message += f"👤 <b>Ism:</b> {user_name}\n"
        message += f"📞 <b>Telefon:</b> {phone}\n"

        # Section: Logistics
        if lead.address_text or (lead.latitude and lead.longitude):
            message += "\n📍 <b>LOGISTIKA</b>\n"
            if lead.latitude is not None and lead.longitude is not None:
                message += (
                    f"📍 <b>Lokatsiya:</b> "
                    f"<a href='https://maps.google.com/?q={lead.latitude},{lead.longitude}'>"
                    f"Xaritada ochish</a>\n"
                )
            if lead.address_text:
                message += f"🏠 <b>Manzil:</b> {lead.address_text}\n"

        # Section: Additional Info
        if lead.message:
            message += f"\n📝 <b>Izoh:</b> {lead.message}\n"

        message += f"\n🚀 <a href='https://tanla-ai.ardentsoft.uz/adminka/leads/{lead.id}'>Admin panelda ochish</a>"

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

        # Extra: if geo coords were provided, also drop a native map pin so the
        # recipient can navigate to the address with one tap.
        if lead.latitude is not None and lead.longitude is not None:
            NotificationService.send_telegram_location(lead.latitude, lead.longitude)

        # Notify Company Owner if applicable
        if lead.company and lead.company.user and lead.company.user.telegram_id:
            company_chat_id = str(lead.company.user.telegram_id)
            message_company = message + "\n\n<i>Sizning kompaniyangizga yangi so'rov tushdi!</i>"
            NotificationService.send_telegram_message(
                message_company, chat_id=company_chat_id, reply_markup=reply_markup
            )
            if lead.latitude is not None and lead.longitude is not None:
                NotificationService.send_telegram_location(
                    lead.latitude, lead.longitude, chat_id=company_chat_id
                )

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

    @staticmethod
    def notify_company_created(company):
        """Admin gets a ping when a new company is created."""
        owner = company.user
        owner_name = f"{owner.first_name or ''} {owner.last_name or ''}".strip()
        
        message = (
            "🏢 <b>YANGI KAMPANIYA YARATILDI</b>\n\n"
            f"🏷️ <b>Nomi:</b> {company.name}\n"
            f"👤 <b>Egasi:</b> {owner_name}\n"
            f"📞 <b>Telefon:</b> {company.phone or '—'}\n"
            f"📍 <b>Manzil:</b> {company.location or '—'}\n\n"
            "⏳ Status: <b>PENDING</b> (To'lov kutilmoqda)\n"
        )
        
        message += (
            f"\n🚀 <a href='https://tanla-ai.ardentsoft.uz/adminka/companies/"
            f"{company.id}'>Admin panelda ko'rish</a>"
        )
        
        NotificationService.send_telegram_message(message)


    # ── Payment notifications ───────────────────────────────
    # Each of these is called from PaymentViewSet / AdminPaymentViewSet.
    # Kept as static methods on NotificationService for consistency with the
    # existing `notify_new_lead` / `notify_lead_reminder` style.

    @staticmethod
    def notify_payment_submitted(payment):
        """Admin gets a ping when an owner uploads a payment screenshot."""
        company = payment.company
        owner = company.user
        owner_name = (
            f"{owner.first_name or ''} {owner.last_name or ''}".strip()
            if owner else "?"
        )
        try:
            amount_fmt = f"{float(payment.amount):,.0f}".replace(",", " ")
        except Exception:
            amount_fmt = str(payment.amount)

        message = (
            "💳 <b>YANGI TO'LOV TASDIQLASH SO'ROVI</b>\n\n"
            f"🏢 <b>Kompaniya:</b> {company.name}\n"
            f"👤 <b>Egasi:</b> {owner_name}\n"
            f"💰 <b>Summa:</b> {amount_fmt} so'm\n"
            f"📅 <b>Muddati:</b> {payment.months} oy\n"
        )
        if payment.note:
            message += f"📝 <b>Izoh:</b> {payment.note}\n"

        message += (
            f"\n🚀 <a href='https://tanla-ai.ardentsoft.uz/adminka/payments/"
            f"{payment.id}'>Admin panelda ko'rish</a>"
        )

        reply_markup = {
            "inline_keyboard": [[
                {"text": "✅ Tasdiqlash", "callback_data": f"pay_approve_{payment.id}"},
                {"text": "❌ Rad etish",  "callback_data": f"pay_reject_{payment.id}"},
            ]]
        }
        NotificationService.send_telegram_message(message, reply_markup=reply_markup)

    @staticmethod
    def notify_payment_approved(payment, reactivated_count=0):
        """Owner gets a confirmation with new deadline + reactivated count."""
        company = payment.company
        if not (company.user and company.user.telegram_id):
            return

        deadline = company.subscription_deadline
        deadline_str = deadline.strftime("%Y-%m-%d") if deadline else "?"

        message = (
            "✅ <b>To'lov tasdiqlandi!</b>\n\n"
            f"🏢 <b>Kompaniya:</b> {company.name}\n"
            f"📅 <b>Yangi muddat:</b> {deadline_str}\n"
            f"➕ <b>{payment.months} oy</b> qo'shildi.\n"
        )
        if reactivated_count:
            message += (
                f"\n♻️ <b>{reactivated_count} ta mahsulot</b> qayta faollashtirildi."
            )

        NotificationService.send_telegram_message(
            message, chat_id=str(company.user.telegram_id)
        )

    @staticmethod
    def notify_payment_rejected(payment):
        """Owner gets the rejection reason so they can resubmit."""
        company = payment.company
        if not (company.user and company.user.telegram_id):
            return

        message = (
            "❌ <b>To'lov rad etildi</b>\n\n"
            f"🏢 <b>Kompaniya:</b> {company.name}\n"
            f"📝 <b>Sabab:</b> {payment.rejection_reason or '—'}\n\n"
            "Iltimos, to'lovni qayta yuboring."
        )
        NotificationService.send_telegram_message(
            message, chat_id=str(company.user.telegram_id)
        )

    @staticmethod
    def notify_subscription_expiring(company, days_left):
        """
        Used by the notify_expiring_subscriptions cron to warn an owner that
        their subscription ends soon. Idempotent at the caller's level — the
        cron decides when (e.g. at 3 days and at 1 day).
        """
        if not (company.user and company.user.telegram_id):
            return

        deadline_str = (
            company.subscription_deadline.strftime("%Y-%m-%d")
            if company.subscription_deadline else "?"
        )
        message = (
            "⏰ <b>Obuna muddati yaqinlashdi</b>\n\n"
            f"🏢 <b>Kompaniya:</b> {company.name}\n"
            f"📅 <b>Tugash sanasi:</b> {deadline_str}\n"
            f"⏳ <b>Qolgan:</b> {days_left} kun\n\n"
            "Listinglar yo'qolib qolmasligi uchun to'lovni oldindan yuboring."
        )
        NotificationService.send_telegram_message(
            message, chat_id=str(company.user.telegram_id)
        )

    # ── Promotion Broadcast ─────────────────────────────────
    @staticmethod
    def send_telegram_photo(photo_url: str, caption: str, chat_id: str = None, reply_markup: dict = None):
        """Sends a photo message to a Telegram chat."""
        token = getattr(settings, 'TELEGRAM_BOT_TOKEN', None)
        target_chat_id = chat_id or getattr(settings, 'ADMIN_TELEGRAM_ID', None)
        if not token or not target_chat_id:
            return False

        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        payload = {
            "chat_id": target_chat_id,
            "photo": photo_url,
            "caption": caption,
            "parse_mode": "HTML",
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram photo: {e}")
            return False

    @staticmethod
    def broadcast_promotion(product, excluded_chat_ids=None):
        """
        Sends a promotion broadcast to ALL Telegram users.
        Returns (sent_count, fail_count).
        """
        from .models import TelegramUser

        excluded = set(excluded_chat_ids or [])
        users = TelegramUser.objects.exclude(telegram_id__in=excluded).values_list('telegram_id', flat=True)

        # Format prices
        try:
            old_price = f"{float(product.price):,.0f}".replace(",", " ") if product.price else "—"
        except Exception:
            old_price = str(product.price) if product.price else "—"
        try:
            new_price = f"{float(product.discount_price):,.0f}".replace(",", " ") if product.discount_price else "—"
        except Exception:
            new_price = str(product.discount_price) if product.discount_price else "—"

        # Discount percentage
        discount_pct = ""
        try:
            if product.price and product.discount_price:
                pct = round((1 - float(product.discount_price) / float(product.price)) * 100)
                discount_pct = f" (-{pct}%)"
        except Exception:
            pass

        company_name = product.company.name if product.company else "Tanla"
        end_date = ""
        if product.sale_end_date:
            try:
                from django.utils import timezone as tz
                end_date = f"\n⏳ <b>Aksiya muddati:</b> {product.sale_end_date.strftime('%d.%m.%Y')}"
            except Exception:
                pass

        caption = (
            f"🔥 <b>AKSIYA!</b>{discount_pct}\n\n"
            f"🚪 <b>{product.name}</b>\n"
            f"🏢 {company_name}\n\n"
            f"💰 <s>{old_price} so'm</s> → <b>{new_price} so'm</b>\n"
            f"{end_date}\n\n"
            "📲 Batafsil ko'rish uchun ilovani oching!"
        )

        # Try to get absolute image URL
        image_url = None
        if product.image_no_bg:
            img = product.image_no_bg.url
        elif product.image:
            img = product.image.url
        else:
            img = None

        if img:
            backend_url = getattr(settings, 'BACKEND_URL', 'https://tanla-ai.ardentsoft.uz').rstrip('/')
            if img.startswith('http'):
                image_url = img
            else:
                image_url = f"{backend_url}{img}"

        sent = 0
        failed = 0
        for tg_id in users:
            chat_id = str(tg_id)
            try:
                if image_url:
                    ok = NotificationService.send_telegram_photo(image_url, caption, chat_id=chat_id)
                else:
                    ok = NotificationService.send_telegram_message(caption, chat_id=chat_id)
                if ok:
                    sent += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

        logger.info(f"Promotion broadcast: sent={sent}, failed={failed}, product={product.id}")
        return sent, failed

