import logging
import os
import shlex
import subprocess
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.db import models
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import parsers, permissions, status, views, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied, ValidationError
from rest_framework.response import Response

from ..models import (
    AIResult,
    Category,
    Company,
    HomeBanner,
    LeadRequest,
    Payment,
    Product,
    ProductImage,
    Subscription,
    SystemSettings,
    TelegramUser,
    Wishlist,
)
from ..utils import verify_telegram_webapp_data
from .serializers import (
    AIResultSerializer,
    CategorySerializer,
    CompanySerializer,
    HomeBannerSerializer,
    LeadRequestSerializer,
    PaymentSerializer,
    ProductSerializer,
    TelegramUserSerializer,
    WishlistSerializer,
)

ai_executor = ThreadPoolExecutor(max_workers=2)


def format_generation_error(error):
    message = str(error or "").strip()
    if not message:
        return "AI generation failed."

    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return "AI generation failed."

    short_message = lines[-1]
    return short_message[:300]


def run_api_ai_background(
    session_key, session_data_key, product_id, room_path, result_path, tg_user_id, request_id=""
):
    print(f"DEBUG: [AI Service] Background task STARTED for product {product_id}")
    from django.contrib.sessions.backends.db import SessionStore

    session = SessionStore(session_key=session_key)
    try:
        product = Product.objects.get(pk=product_id)
        from ..models import SystemSettings
        from ..services import AIService

        provider = (
            str(getattr(SystemSettings.get_solo(), "ai_provider", "hybrid") or "hybrid")
            .strip()
            .lower()
        )
        print(f"DEBUG: [AI Service] Provider = '{provider}' for product {product_id}")
        print(f"DEBUG: [AI Service] Room path: {room_path}")
        print(f"DEBUG: [AI Service] Result path: {result_path}")

        import os

        if not os.path.exists(room_path):
            raise FileNotFoundError(f"Room image not found: {room_path}")
        room_size = os.path.getsize(room_path)
        print(f"DEBUG: [AI Service] Room image size: {room_size} bytes")

        AIService.generate_room_preview(product, room_path, result_path)

        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Result image was NOT created: {result_path}")
        result_size = os.path.getsize(result_path)
        print(
            f"DEBUG: [AI Service] ✅ Result created: {result_path} ({result_size} bytes)"
        )

        ai_result = None
        if tg_user_id:
            user = TelegramUser.objects.filter(id=tg_user_id).first()
            if user:
                ai_result = AIResult.objects.create(
                    user=user,
                    product=product,
                    image=os.path.join("ai_results", os.path.basename(result_path)),
                    input_image=os.path.join("ai_temp", os.path.basename(room_path)),
                    status="done",
                )
                
                # Upload to Telegram as permanent storage
                from ..notifications import NotificationService
                import datetime
                company_name = product.company.name if product.company else "Tanla AI"
                user_details = f"{user.first_name or ''} {user.last_name or ''}".strip()
                if user.username:
                    user_details += f" (@{user.username})"
                user_details = user_details or "Noma'lum"

                caption = (
                    f"🆔 User ID: {user.id}\n"
                    f"👤 Mijoz: {user_details}\n"
                    f"🏢 Kompaniya: {company_name}\n"
                    f"🕒 Vaqt: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                    f"🚪 Eshik: {product.name} (ID: {product.id})\n"
                    f"🔗 Havola: https://tanla-ai.ardentsoft.uz/product/{product.id}\n\n"
                    f"#tanlaai #result"
                )
                file_ids = NotificationService.send_media_group_to_telegram(
                    [room_path, result_path], 
                    caption
                )
                if file_ids:
                    # The last image in the group is our result path
                    ai_result.telegram_file_id = file_ids[-1]
                    ai_result.save(update_fields=['telegram_file_id'])

                # Automatically create a "soft" lead if enabled in system settings
                from ..models import SystemSettings

                if SystemSettings.get_solo().auto_create_lead:
                    LeadRequest.objects.create(
                        user=user,
                        product=product,
                        company=product.company,
                        ai_result=ai_result,
                        lead_type="visualize",
                        status="new",
                        phone=user.phone or "",
                        message="Mijoz mahsulotni SI orqali vizualizatsiya qildi.",
                    )

        data = session.get(session_data_key, {})
        data["status"] = "done"
        if ai_result:
            data["ai_result_id"] = ai_result.id
        session[session_data_key] = data
        
        # Web App Cache fallback for broken sessions
        if tg_user_id and request_id:
            from django.core.cache import cache
            cache.set(f"ai_job_user_{tg_user_id}_req_{request_id}", {"status": "done", "ai_result_id": ai_result.id if ai_result else None}, timeout=3600)
            
        logger.info(
            f"DEBUG: [AI Service] Background task COMPLETED for product {product_id}"
        )
    except Exception as error:
        error_msg = traceback.format_exc()
        print(f"ERROR: [AI Service] Background task FAILED for product {product_id}")
        print(f"ERROR: [AI Service] Exception type: {type(error).__name__}")
        print(f"ERROR: [AI Service] Message: {str(error)[:500]}")
        print(f"ERROR: [AI Service] Traceback:\n{error_msg}")
        # Write to dedicated debug log for easy tailing
        try:
            import time

            with open("ai_debug.log", "a") as _f:
                _f.write(
                    f"\n--- ERROR product={product_id} [{time.ctime()}] ---\n"
                    f"provider attempted: see DEBUG lines above\n"
                    f"{error_msg}\n"
                )
        except Exception:
            pass
        Product.objects.filter(pk=product_id).update(ai_error=error_msg[:1000])
        data = session.get(session_data_key, {})
        data["status"] = "error"
        data["error_msg"] = format_generation_error(error)
        session[session_data_key] = data
        
        if tg_user_id and request_id:
            from django.core.cache import cache
            cache.set(f"ai_job_user_{tg_user_id}_req_{request_id}", {"status": "error", "error_msg": format_generation_error(error)}, timeout=3600)
            
        logger.error(
            f"DEBUG: API AI generation error for product {product_id}: {error}"
        )
    finally:
        session.save()


def build_ai_result_payload(request, ai_result):
    from ..ai_utils import load_visualization_metadata
    from django.urls import reverse

    if ai_result.image:
        img_url = request.build_absolute_uri(ai_result.image.url)
    elif ai_result.telegram_file_id:
        # Avoid hardcoding by using reverse
        proxy_path = reverse('api_telegram_proxy', kwargs={'file_id': ai_result.telegram_file_id})
        img_url = request.build_absolute_uri(proxy_path)
    else:
        img_url = ""

    payload = {
        "id": ai_result.id,
        "status": "done",
        "image_url": img_url,
    }

    metadata = None
    try:
        if ai_result.image:
            metadata = load_visualization_metadata(ai_result.image.path)
    except Exception:
        metadata = None

    if metadata:
        payload["analysis"] = metadata.get("analysis")
        payload["generation_prompt"] = metadata.get("generation_prompt")
        payload["generation_meta"] = metadata.get("generation_meta")
        payload["pipeline"] = metadata.get("pipeline")

    return payload


def get_tg_user(request):
    tg_user_id = request.session.get("tg_user_id")

    # Fallback to header-based authentication and auto-registration
    if not tg_user_id:
        init_data = request.headers.get("X-Telegram-Init-Data")
        if init_data:
            user_data = verify_telegram_webapp_data(
                init_data, settings.TELEGRAM_BOT_TOKEN
            )
            if user_data:
                # Auto-sync/register user
                user_id = user_data.get("id")
                user, created = TelegramUser.objects.update_or_create(
                    telegram_id=user_id,
                    defaults={
                        "first_name": user_data.get("first_name", ""),
                        "last_name": user_data.get("last_name", ""),
                        "username": user_data.get("username"),
                    },
                )
                tg_user_id = user.id
                # Update session for subsequent requests
                request.session["tg_user_id"] = tg_user_id

    if not tg_user_id:
        return None
    return TelegramUser.objects.filter(id=tg_user_id).first()


def require_tg_user(request):
    tg_user = get_tg_user(request)
    if tg_user is None:
        raise ValidationError({"detail": "Authentication required"})
    return tg_user


def ensure_product_owner(request, product):
    tg_user = require_tg_user(request)
    if request.user.is_staff or product.owner_id == tg_user.id:
        return tg_user
    raise PermissionDenied("You do not have permission to manage this product.")


def ensure_company_owner(request, company):
    tg_user = require_tg_user(request)
    if request.user.is_staff or company.user_id == tg_user.id:
        return tg_user
    raise PermissionDenied("You do not have permission to manage this company.")


class TelegramAuthView(views.APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        if settings.DEBUG:
            # Mock login for local browser testing
            user_data = {
                "id": 123456789,
                "first_name": "Test",
                "last_name": "User",
                "username": "testuser",
                "photo_url": "https://via.placeholder.com/150",
            }
            return self._process_user(request, user_data)
        return Response(
            {"error": "Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED
        )

    def post(self, request):
        init_data = request.data.get("initData")
        if not init_data:
            return Response(
                {"error": "No initData provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        user_data = verify_telegram_webapp_data(init_data, settings.TELEGRAM_BOT_TOKEN)
        if not user_data:
            return Response(
                {"error": "Invalid Telegram data"}, status=status.HTTP_403_FORBIDDEN
            )

        return self._process_user(request, user_data)

    def _process_user(self, request, user_data):
        telegram_id = user_data.get("id")
        first_name = user_data.get("first_name")
        last_name = user_data.get("last_name")
        username = user_data.get("username")
        photo_url = user_data.get("photo_url")

        user, created = TelegramUser.objects.update_or_create(
            telegram_id=telegram_id,
            defaults={
                "first_name": first_name,
                "last_name": last_name,
                "username": username,
                "photo_url": photo_url,
            },
        )

        # We'll use DRF sessions for now as it's easier to integrate with existing logic
        request.session["tg_user_id"] = user.id
        request.session.modified = True

        serializer = TelegramUserSerializer(user)
        return Response({"user": serializer.data, "status": "ok"})


class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Category.objects.annotate(
        product_count=models.Count("products")
    ).order_by("id")
    serializer_class = CategorySerializer
    permission_classes = [permissions.AllowAny]


class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    permission_classes = [permissions.AllowAny]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser, parsers.JSONParser]

    def get_queryset(self):
        queryset = Product.objects.select_related("category", "company", "owner").all()
        category = self.request.query_params.get("category")
        company = self.request.query_params.get("company")
        is_featured = self.request.query_params.get("is_featured")
        is_on_sale = self.request.query_params.get("is_on_sale")
        search = self.request.query_params.get("search")
        now = timezone.now()

        if self.action == "list":
            # Show products that either have no company (system products)
            # OR belong to an active company (VIP or with a valid subscription).
            # We filter out blocked companies.
            queryset = queryset.filter(is_active=True).filter(
                models.Q(company__isnull=True)
                | (
                    ~models.Q(company__status='blocked')
                    & (
                        models.Q(company__is_vip=True)
                        | (
                            models.Q(company__is_active=True)
                            & models.Q(company__status='active')
                            & models.Q(company__subscription_deadline__gt=now)
                        )
                    )
                )
            )

        if category:
            queryset = queryset.filter(category_id=category)
        if company:
            queryset = queryset.filter(company_id=company)
        if is_featured:
            queryset = queryset.filter(is_featured=True)
        if is_on_sale:
            # Handle string boolean values from query params
            is_on_sale_bool = str(is_on_sale).lower() in ("true", "1", "yes")
            if is_on_sale_bool:
                queryset = queryset.filter(is_on_sale=True).filter(
                    models.Q(sale_end_date__gt=now)
                    | models.Q(sale_end_date__isnull=True)
                )
            else:
                queryset = queryset.filter(is_on_sale=False)

        if search:
            queryset = queryset.filter(
                models.Q(name__icontains=search)
                | models.Q(description__icontains=search)
            )

        is_wishlisted = self.request.query_params.get("is_wishlisted")
        if is_wishlisted:
            is_wishlisted_bool = str(is_wishlisted).lower() in ("true", "1", "yes")
            if is_wishlisted_bool:
                tg_user = get_tg_user(self.request)
                if tg_user:
                    queryset = queryset.filter(wishlisted_by__user_id=tg_user.id)
                else:
                    queryset = queryset.none()

        return queryset.order_by("-id")

    @action(detail=False, methods=["get"])
    def my(self, request):
        tg_user = get_tg_user(request)
        if tg_user is None:
            return Response(
                {"error": "Not authenticated"}, status=status.HTTP_401_UNAUTHORIZED
            )

        products = Product.objects.filter(owner_id=tg_user.id).select_related(
            "category", "company", "owner"
        )
        serializer = self.get_serializer(products, many=True)
        return Response(serializer.data)

    # ---- Gallery helpers (ProductImage: 1 cutout + up to 4 showcase) ----
    def _sync_product_gallery(self, request, product):
        """
        Reads multipart fields from the request and syncs ProductImage rows
        for the given product. Fields handled (all optional):
          - gallery_images: one or more uploaded files (FILE list)
          - gallery_main_index: int — which of the new uploads is the cutout
          - delete_image_ids: comma-separated IDs to remove
        Silently ignores missing fields so this stays backwards-compatible
        with existing create/update calls that don't send any gallery data.
        """
        # 1) Deletions
        delete_ids = (request.data.get("delete_image_ids") or "").strip()
        if delete_ids:
            ids = [int(x) for x in delete_ids.split(",") if x.strip().isdigit()]
            if ids:
                ProductImage.objects.filter(product=product, id__in=ids).delete()

        # 2) New uploads
        new_files = request.FILES.getlist("gallery_images")
        if not new_files:
            return

        # Enforce MAX_IMAGES_PER_PRODUCT across existing + new
        existing_count = product.images.count()
        allowed = ProductImage.MAX_IMAGES_PER_PRODUCT - existing_count
        if allowed <= 0:
            raise ValidationError(
                {"gallery_images": [
                    f"Max {ProductImage.MAX_IMAGES_PER_PRODUCT} ta rasm ruxsat etilgan."
                ]}
            )
        if len(new_files) > allowed:
            raise ValidationError(
                {"gallery_images": [
                    f"Faqat {allowed} ta rasm qo'shish mumkin "
                    f"(jami limit: {ProductImage.MAX_IMAGES_PER_PRODUCT})."
                ]}
            )

        raw_main = request.data.get("gallery_main_index", None)
        try:
            main_index = int(raw_main) if raw_main not in (None, "") else None
        except (TypeError, ValueError):
            main_index = None

        has_existing_main = product.images.filter(is_main=True).exists()

        # If product has no main yet AND client didn't tell us which new upload
        # is the cutout, refuse — otherwise we'd silently pick the first file
        # which leads to confusing bugs for the company user.
        if not has_existing_main and main_index is None:
            raise ValidationError(
                {"gallery_main_index": [
                    "Asosiy (foni olingan) rasmni tanlang."
                ]}
            )

        if main_index is not None and not (0 <= main_index < len(new_files)):
            raise ValidationError(
                {"gallery_main_index": [
                    "gallery_main_index diapazondan tashqarida."
                ]}
            )

        next_order = (product.images.aggregate(
            m=models.Max("order")
        )["m"] or 0) + 1

        for i, f in enumerate(new_files):
            ProductImage.objects.create(
                product=product,
                image=f,
                is_main=(i == main_index),
                order=next_order + i,
            )

    def perform_create(self, serializer):
        tg_user = require_tg_user(self.request)
        company = getattr(tg_user, "company", None)
        if company is None:
            raise ValidationError(
                {"detail": "Create a company profile before adding products."}
            )
        if company.status != 'active':
            raise PermissionDenied(
                "Do'koningiz hali aktivlashtirilmagan. Mahsulot qo'shish uchun avval to'lovni tasdiqlang."
            )

        subscription, _ = Subscription.objects.get_or_create(company=company)
        current_count = company.products.count()
        if current_count >= subscription.max_products:
            # Structured payload so the frontend can render a real "limit reached"
            # screen with an upgrade CTA, not just a toast. Keep `detail` for
            # backward-compat with existing alert handlers.
            raise ValidationError(
                {
                    "detail": (
                        f"Mahsulot limiti tugadi: {current_count}/"
                        f"{subscription.max_products}. Tarifni yangilang."
                    ),
                    "code": "product_limit_reached",
                    "current": current_count,
                    "limit": subscription.max_products,
                }
            )

        product = serializer.save(owner=tg_user, company=company, is_active=True)
        self._sync_product_gallery(self.request, product)

    def perform_update(self, serializer):
        ensure_product_owner(self.request, serializer.instance)
        product = serializer.save()
        self._sync_product_gallery(self.request, product)

    def perform_destroy(self, instance):
        ensure_product_owner(self.request, instance)
        instance.delete()

    @action(detail=True, methods=["post"], url_path="toggle-active")
    def toggle_active(self, request, pk=None):
        """
        Owner-only: pause/resume this product listing without deleting it.
        Returns the new state so the frontend can optimistic-update in place.
        Staff users can also toggle (for moderation).
        """
        product = self.get_object()
        ensure_product_owner(request, product)
        product.is_active = not product.is_active
        product.save(update_fields=["is_active"])
        return Response(
            {
                "id": product.id,
                "is_active": product.is_active,
            }
        )

    @action(detail=True, methods=["post"], permission_classes=[permissions.AllowAny])
    def toggle_wishlist(self, request, pk=None):
        product = self.get_object()
        tg_user = get_tg_user(request)
        if tg_user is None:
            return Response(
                {"error": "Not authenticated"}, status=status.HTTP_401_UNAUTHORIZED
            )

        wishlist_item = Wishlist.objects.filter(user=tg_user, product=product).first()

        if wishlist_item:
            wishlist_item.delete()
            return Response({"status": "removed"})
        else:
            Wishlist.objects.create(user=tg_user, product=product)
            return Response({"status": "added"})

    @action(detail=True, methods=["post"], permission_classes=[permissions.AllowAny])
    def reprocess_ai(self, request, pk=None):
        """Forces the AI to re-process the background removal using the latest model (SAM)."""
        product = self.get_object()
        product.ai_status = "processing"
        product.save(update_fields=["ai_status"])

        import threading

        # Delayed import to prevent global API crash if dependencies are missing
        from ..services import AIService

        threading.Thread(
            target=AIService.process_product_background, args=(product,)
        ).start()

        return Response({"status": "processing"})

    @action(
        detail=True,
        methods=["post"],
        url_path="ai-generate",
        permission_classes=[permissions.AllowAny],
    )
    def ai_generate(self, request, pk=None):
        product = self.get_object()
        tg_user = get_tg_user(request)

        if tg_user is None:
            return Response(
                {"status": "error", "message": "Authentication required"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        # Allow admin-added products without a company
        if product.company:
            subscription, _ = Subscription.objects.get_or_create(
                company=product.company
            )
            current_ai_count = AIResult.objects.filter(
                product__company=product.company, status="done"
            ).count()
            if current_ai_count >= subscription.ai_generations_limit:
                return Response(
                    {
                        "status": "error",
                        "message": f"AI generation limit reached ({subscription.ai_generations_limit}).",
                        "code": "limit_reached",
                        "limit": subscription.ai_generations_limit,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        if product.ai_status == "processing":
            return Response(
                {
                    "status": "preparing",
                    "message": "Mahsulot rasmi hali tayyorlanmoqda. Bir necha soniyadan keyin qayta urinib ko‘ring.",
                    "code": "not_ready",
                }
            )

        # If background is not removed yet, trigger it now instead of failing
        if product.ai_status == "none":
            from ..models import SystemSettings

            provider = (
                str(
                    getattr(SystemSettings.get_solo(), "ai_provider", "hybrid")
                    or "hybrid"
                )
                .strip()
                .lower()
            )

            if provider == "gemini_direct":
                # Gemini Direct uchun background removal shart emas — original rasmni ishlatadi
                pass
            else:
                # Boshqa providerlar uchun background removal kerak
                product.ai_status = (
                    "processing"  # Set to processing to avoid double trigger
                )
                product.save(update_fields=["ai_status"])

                # Use the service to start background removal
                # Delayed import to prevent global API crash if dependencies are missing
                import threading

                from ..services import AIService

                threading.Thread(
                    target=AIService.process_product_background, args=(product,)
                ).start()

                return Response(
                    {
                        "status": "preparing",
                        "message": "Orqa fon o'chirilmoqda. Iltimos 10-15 soniya kutib qayta urinib ko'ring.",
                        "code": "not_ready",
                    },
                    status=status.HTTP_200_OK,
                )  # Return 200 instead of 400 for better UX

        room_photo = request.FILES.get("room_photo") or request.FILES.get("input_image")
        user_height = str(request.data.get("height", "")).strip()
        user_width = str(request.data.get("width", "")).strip()

        if not room_photo:
            return Response(
                {
                    "status": "error",
                    "message": "Please upload a room photo.",
                    "code": "no_photo",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if product.height and product.width and user_height and user_width:
            tolerance = 5.0
            try:
                if (
                    abs(float(user_height) - float(product.height)) > tolerance
                    or abs(float(user_width) - float(product.width)) > tolerance
                ):
                    return Response(
                        {
                            "status": "error",
                            "message": "Door o‘lchamlari tanlangan mahsulotga mos kelmayapti.",
                            "code": "dimension_mismatch",
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )
            except (TypeError, ValueError):
                pass  # Fallback if dimensions are invalidly formatted

        request_id = str(uuid.uuid4())
        room_dir = os.path.join(settings.MEDIA_ROOT, "ai_temp")
        result_dir = os.path.join(settings.MEDIA_ROOT, "ai_results")
        os.makedirs(room_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        room_path = os.path.join(room_dir, f"{request_id}.png")
        result_path = os.path.join(result_dir, f"{request_id}.png")

        with open(room_path, "wb+") as destination:
            for chunk in room_photo.chunks():
                destination.write(chunk)

        if not request.session.session_key:
            request.session.create()

        session_key_data = f"ai_gen_{product.id}"
        request.session[session_key_data] = {
            "status": "running",
            "request_id": request_id,
            "product_id": product.id,
        }
        request.session.modified = True
        request.session.save()
        
        from django.core.cache import cache
        cache.set(f"ai_job_user_{tg_user.id}_req_{request_id}", {"status": "running"}, timeout=3600)

        print(
            f"DEBUG: [AI Service] Submitting background task for product {product.id}"
        )
        ai_executor.submit(
            run_api_ai_background,
            request.session.session_key,
            session_key_data,
            product.id,
            room_path,
            result_path,
            tg_user.id,
            request_id,
        )
        print(f"DEBUG: [AI Service] Task submitted successfully")

        return Response({"status": "ok", "request_id": request_id})

    @action(
        detail=True,
        methods=["get"],
        url_path="ai-generate/result",
        permission_classes=[permissions.AllowAny],
    )
    def ai_generate_result(self, request, pk=None):
        product = self.get_object()
        tg_user = get_tg_user(request)
        session_key_data = f"ai_gen_{product.id}"
        session_data = request.session.get(session_key_data, {})
        request_id = str(request.query_params.get("request_id", "")).strip()

        # Build response strictly to avoid client-side caching
        def no_store(res):
            res["Cache-Control"] = "no-store, max-age=0, must-revalidate"
            res["Pragma"] = "no-cache"
            return res

        # Reliable check for telegram webview cross-site blocking
        if tg_user and request_id:
            from django.core.cache import cache
            job_data = cache.get(f"ai_job_user_{tg_user.id}_req_{request_id}")
            if job_data:
                jstatus = job_data.get("status")
                if jstatus == "done":
                    # Clear it so subsequent totally fresh entries don't instantly resolve
                    cache.delete(f"ai_job_user_{tg_user.id}_req_{request_id}")
                    ai_res_id = job_data.get("ai_result_id")
                    if ai_res_id:
                        ai_res = AIResult.objects.filter(id=ai_res_id).first()
                        if ai_res:
                            return no_store(Response(build_ai_result_payload(request, ai_res)))
                    return no_store(Response({"status": "done"}))
                elif jstatus == "error":
                    cache.delete(f"ai_job_user_{tg_user.id}_req_{request_id}")
                    return no_store(Response({"status": "error", "message": job_data.get("error_msg", "Generatsiya bekor qilindi")}))
                elif jstatus in ("running", "processing"):
                    return no_store(Response({"status": "processing"}))

        if session_data:
            current_request_id = str(session_data.get("request_id", "")).strip()
            if request_id and current_request_id and request_id != current_request_id:
                return no_store(Response({"status": "processing"}))

            session_status = session_data.get("status")
            if session_status == "done":
                ai_result_id = session_data.get("ai_result_id")
                if ai_result_id:
                    ai_result = AIResult.objects.filter(id=ai_result_id).first()
                    if ai_result:
                        return no_store(Response(build_ai_result_payload(request, ai_result)))
                return no_store(Response({"status": "done"}))

            elif session_status == "error":
                return no_store(Response(
                    {
                        "status": "error",
                        "message": session_data.get("error_msg")
                        or "AI generation failed.",
                    }
                ))

            elif session_status in {"running", "processing", "pending"}:
                return no_store(Response({"status": "processing"}))

        # DB Fallback: ONLY when there's no request_id (page reload after result was ready)
        # When request_id IS present, we're actively polling for a specific upload — 
        # never return old results, just keep saying "processing"
        if not request_id and tg_user:
            latest_result = (
                AIResult.objects.filter(product=product, user=tg_user)
                .order_by("-created_at")
                .first()
            )
            if latest_result:
                return no_store(Response(build_ai_result_payload(request, latest_result)))

        if not request_id and product.ai_status == "error":
            return no_store(Response(
                {
                    "status": "error",
                    "message": format_generation_error(product.ai_error),
                }
            ))

        return no_store(Response({"status": "processing"}))


class CompanyViewSet(viewsets.ModelViewSet):
    queryset = Company.objects.all()
    serializer_class = CompanySerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        if self.action == "list":
            query = self.request.query_params.get(
                "search"
            ) or self.request.query_params.get("q")
            now = timezone.now()
            companies = (
                Company.objects.filter(
                    ~models.Q(status='blocked')
                ).filter(
                    models.Q(is_vip=True)
                    | (
                        models.Q(is_active=True)
                        & models.Q(status='active')
                        & models.Q(subscription_deadline__gt=now)
                    )
                )
                .annotate(
                    product_count=models.Count("products", distinct=True),
                    ai_usage=models.Count("products__ai_visualizations", distinct=True),
                    wishlist_count=models.Count(
                        "products__wishlisted_by", distinct=True
                    ),
                )
                .annotate(
                    score=models.F("product_count")
                    + models.F("ai_usage")
                    + models.F("wishlist_count")
                )
            )
            if query:
                companies = companies.filter(
                    models.Q(name__icontains=query)
                    | models.Q(location__icontains=query)
                )
            return companies.order_by("-score", "-created_at")
        return Company.objects.select_related("user").all()

    def perform_create(self, serializer):
        tg_user = require_tg_user(self.request)
        if hasattr(tg_user, "company"):
            raise ValidationError({"detail": "You already have a company profile."})

        # Create company
        try:
            company = serializer.save(user=tg_user)
        except Exception as e:
            print(f"DEBUG: Company creation error: {e}")
            raise

        # Upgrade user role to COMPANY automatically
        if tg_user.role != "COMPANY":
            tg_user.role = "COMPANY"
            tg_user.save(update_fields=["role"])

        Subscription.objects.get_or_create(company=company)

        # Notify admin
        from ..notifications import NotificationService
        NotificationService.notify_company_created(company)

    def perform_update(self, serializer):
        ensure_company_owner(self.request, serializer.instance)
        serializer.save()

    def perform_destroy(self, instance):
        ensure_company_owner(self.request, instance)
        instance.delete()

    @action(detail=False, methods=['get'])
    def leaderboard(self, request):
        """
        Returns top companies ranked by the number of successful conversions.
        """
        now = timezone.now()
        companies = (
            Company.objects.filter(
                ~models.Q(status='blocked')
            ).filter(
                models.Q(is_vip=True)
                | (
                    models.Q(is_active=True)
                    & models.Q(status='active')
                    & models.Q(subscription_deadline__gt=now)
                )
            ).annotate(
                converted_leads=models.Count('leads', filter=models.Q(leads__status='converted'), distinct=True),
                total_leads=models.Count('leads', distinct=True),
                ai_usage=models.Count("products__ai_visualizations", distinct=True),
            ).order_by('-converted_leads', '-total_leads')[:20]
        )
        
        serializer = CompanySerializer(companies, many=True, context={'request': request})
        return Response(serializer.data)

    @action(detail=False, methods=["get", "post", "put", "patch"])
    def my(self, request):
        tg_user = get_tg_user(request)
        if tg_user is None:
            return Response(
                {"error": "Not authenticated"}, status=status.HTTP_401_UNAUTHORIZED
            )

        company = Company.objects.filter(user_id=tg_user.id).first()

        if request.method == "GET":
            if not company:
                return Response(
                    {"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND
                )
            serializer = self.get_serializer(company)
            return Response(serializer.data)

        partial = request.method == "PATCH"
        serializer = self.get_serializer(company, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        serializer.save(user=tg_user)
        return Response(
            serializer.data,
            status=status.HTTP_201_CREATED if company is None else status.HTTP_200_OK,
        )

    @action(detail=False, methods=["get"], url_path="my/subscription")
    def my_subscription(self, request):
        """
        Dashboard endpoint: returns current company's product usage and
        subscription window. Frontend uses this to render the "12/30 ta mahsulot
        ishlatilgan" widget + "Obuna X kundan keyin tugaydi" warning.

        We source the deadline from `Company.subscription_deadline` because
        that's the field already used everywhere to filter listings — keeping
        a single source of truth avoids the two-fields drift bug.
        """
        from django.utils import timezone
        from ..models import Subscription

        tg_user = get_tg_user(request)
        if tg_user is None:
            return Response(
                {"error": "Not authenticated"}, status=status.HTTP_401_UNAUTHORIZED
            )

        company = Company.objects.filter(user_id=tg_user.id).first()
        if not company:
            return Response(
                {"error": "Company not found"}, status=status.HTTP_404_NOT_FOUND
            )

        subscription, _ = Subscription.objects.get_or_create(company=company)
        used = company.products.count()
        limit = subscription.max_products
        deadline = company.subscription_deadline
        now = timezone.now()

        days_left = None
        if deadline:
            delta = deadline - now
            # Round up: if there's ANY time left today, count today as a day.
            days_left = max(0, delta.days + (1 if delta.seconds > 0 else 0))

        return Response(
            {
                "plan": subscription.plan,
                "used": used,
                "limit": limit,
                "remaining": max(0, limit - used),
                "is_at_limit": used >= limit,
                "expires_at": deadline.isoformat() if deadline else None,
                "days_left": days_left,
                # `is_active` mirrors Company.is_currently_active — the property
                # the shop listings already use for visibility filtering.
                "is_active": company.is_currently_active,
                "ai_generations_limit": subscription.ai_generations_limit,
            }
        )


class BannerViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = HomeBanner.objects.filter(is_active=True)
    serializer_class = HomeBannerSerializer


class WishlistViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = WishlistSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        tg_user = get_tg_user(self.request)
        if tg_user is None:
            return Wishlist.objects.none()
        return Wishlist.objects.filter(user_id=tg_user.id).select_related(
            "product", "product__category", "product__company"
        )


class LeadRequestViewSet(viewsets.ModelViewSet):
    serializer_class = LeadRequestSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        tg_user = get_tg_user(self.request)
        if tg_user is None:
            return LeadRequest.objects.none()

        # Creator can see leads related to their products or their company
        return (
            LeadRequest.objects.filter(
                models.Q(product__owner_id=tg_user.id)
                | models.Q(company__user_id=tg_user.id)
            )
            .distinct()
            .select_related("product", "user", "company")
        )

    def perform_create(self, serializer):
        tg_user = require_tg_user(self.request)
        product = serializer.validated_data["product"]
        phone = serializer.validated_data.get("phone")

        # Persist phone to user profile for future automated leads
        if phone and not tg_user.phone:
            tg_user.phone = phone
            tg_user.save(update_fields=["phone"])

        # Security: price is ALWAYS computed on the server.
        # Any `calculated_price` sent by the client is discarded so a malicious
        # frontend can't quote itself a $1 door.
        width_cm = serializer.validated_data.get("width_cm")
        height_cm = serializer.validated_data.get("height_cm")
        calculated = None

        if width_cm and height_cm and product.price_per_m2:
            from decimal import Decimal
            try:
                area_m2 = (Decimal(width_cm) * Decimal(height_cm)) / Decimal(10000)
                calculated = (area_m2 * product.price_per_m2).quantize(Decimal("0.01"))
            except Exception:
                calculated = None

        serializer.save(
            user=tg_user,
            company=product.company,
            calculated_price=calculated,
        )


class AIResultViewSet(viewsets.ModelViewSet):
    serializer_class = AIResultSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        tg_user = get_tg_user(self.request)
        if tg_user is None:
            return AIResult.objects.none()
        return (
            AIResult.objects.filter(user_id=tg_user.id)
            .select_related("product", "product__category", "product__company")
            .order_by("-created_at")
        )

    @action(detail=True, methods=["post"], url_path="convert-to-lead")
    def convert_to_lead(self, request, pk=None):
        ai_result = self.get_object()
        tg_user = require_tg_user(request)

        # Check if already has a lead to avoid duplicates
        existing = LeadRequest.objects.filter(
            user=tg_user, ai_result=ai_result
        ).exists()
        if existing:
            return Response(
                {"status": "exists", "message": "Lead already created for this result"}
            )

        # Create Lead
        LeadRequest.objects.create(
            user=tg_user,
            product=ai_result.product,
            company=ai_result.product.company,
            ai_result=ai_result,
            lead_type="visualize",
            status="new",
            phone=tg_user.phone or "",
            message="Mijoz tarix boyicha ushbu vizualizatsiyaga qiziqish bildirdi.",
        )
        return Response({"status": "ok"})

    @action(detail=True, methods=["get"], url_path="download")
    def download(self, request, pk=None):
        """
        Proxies the image file as a download attachment to bypass CORS restrictions.
        Returns a FileResponse with Content-Disposition: attachment.
        """
        from django.http import FileResponse
        import os
        ai_result = self.get_object()
        if not ai_result.image:
             return Response({"error": "Image not found"}, status=404)
        
        # Open the file
        file_handle = open(ai_result.image.path, 'rb')
        filename = f"tanla_ai_{ai_result.id}.png"
        
        response = FileResponse(file_handle, content_type='image/png')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response


class AdminLoginApiView(views.APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []  # Disable CSRF check for login

    def post(self, request):
        username = (request.data.get("username") or "").strip()
        password = request.data.get("password") or ""
        user = authenticate(request, username=username, password=password)
        if not user or not user.is_staff:
            return Response(
                {"status": "error", "message": "Invalid credentials"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        from rest_framework.authtoken.models import Token

        login(request, user)
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"status": "ok", "username": user.username, "token": token.key})


class AdminLogoutApiView(views.APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []  # Disable CSRF check for logout

    def post(self, request):
        logout(request)
        return Response({"status": "ok"})


class AdminMeApiView(views.APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        if not request.user.is_authenticated or not request.user.is_staff:
            return Response(
                {"is_authenticated": False}, status=status.HTTP_401_UNAUTHORIZED
            )
        return Response(
            {
                "is_authenticated": True,
                "username": request.user.username,
            }
        )


class AdminSystemSettingsApiView(views.APIView):
    permission_classes = [permissions.AllowAny]

    def _check_admin(self, request):
        if not request.user.is_authenticated or not request.user.is_staff:
            raise PermissionDenied("Admin authentication required")

    def get(self, request):
        self._check_admin(request)
        setting = SystemSettings.get_solo()

        # Collect all fields dynamically
        data = {}
        for field in setting._meta.fields:
            if field.name not in ["id", "updated_at"]:
                data[field.name] = getattr(setting, field.name)

        # Add deploy status from env
        data["deploy_enabled"] = str(
            getattr(settings, "ALLOW_ADMIN_DEPLOY_ACTIONS", False)
        ).lower() in ("1", "true", "yes", "on")
        return Response(data)

    def post(self, request):
        self._check_admin(request)
        setting = SystemSettings.get_solo()

        # Update allowed fields
        allowed_fields = [
            f.name for f in setting._meta.fields if f.name not in ["id", "updated_at"]
        ]
        updated_fields = []

        for key, value in request.data.items():
            if key in allowed_fields:
                setattr(setting, key, value)
                updated_fields.append(key)

        if updated_fields:
            setting.save(update_fields=updated_fields + ["updated_at"])

        return Response({"status": "ok", "updated": updated_fields})


class AdminBillingApiView(views.APIView):
    """GET/POST for the single-row SystemBilling config."""
    permission_classes = [permissions.AllowAny]

    def _check_admin(self, request):
        if not request.user.is_authenticated or not request.user.is_staff:
            raise PermissionDenied("Admin authentication required")

    def get(self, request):
        self._check_admin(request)
        from ..models import SystemBilling
        billing = SystemBilling.get_solo()
        data = {}
        for field in billing._meta.fields:
            if field.name not in ["id", "updated_at"]:
                val = getattr(billing, field.name)
                # Convert date/decimal to serializable types
                if hasattr(val, 'isoformat'):
                    val = val.isoformat() if val else None
                elif hasattr(val, 'as_tuple'):
                    val = float(val)
                data[field.name] = val
        return Response(data)

    def post(self, request):
        self._check_admin(request)
        from ..models import SystemBilling
        billing = SystemBilling.get_solo()
        allowed_fields = [
            f.name for f in billing._meta.fields if f.name not in ["id", "updated_at"]
        ]
        updated_fields = []
        for key, value in request.data.items():
            if key in allowed_fields:
                # Handle empty date strings
                if key in ('server_due_date', 'ai_due_date') and value == '':
                    value = None
                setattr(billing, key, value)
                updated_fields.append(key)
        if updated_fields:
            billing.save(update_fields=updated_fields + ["updated_at"])
        return Response({"status": "ok", "updated": updated_fields})

class AdminRunActionApiView(views.APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        if not request.user.is_authenticated or not request.user.is_staff:
            return Response(
                {"status": "error", "message": "Admin authentication required"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        deploy_enabled = str(
            getattr(settings, "ALLOW_ADMIN_DEPLOY_ACTIONS", False)
        ).lower() in ("1", "true", "yes", "on")
        if not deploy_enabled:
            return Response(
                {"status": "error", "message": "Deploy actions disabled"},
                status=status.HTTP_403_FORBIDDEN,
            )

        action = request.data.get("action")
        command_map = {
            "git_pull": ["git", "pull", "origin", "main"],
            "migrate": ["python", "manage.py", "migrate"],
            "collectstatic": ["python", "manage.py", "collectstatic", "--noinput"],
            "pip_install": ["pip", "install", "-r", "requirements.txt"],
            "check_models": ["find", "models", "-maxdepth", "3"],
            "restart_service": shlex.split(
                getattr(
                    settings,
                    "ADMIN_RESTART_COMMAND",
                    "sudo systemctl restart tanla-ai.service",
                )
            ),
            "status_service": shlex.split(
                getattr(
                    settings,
                    "ADMIN_STATUS_COMMAND",
                    "sudo systemctl status tanla-ai.service --no-pager",
                )
            ),
        }
        command = command_map.get(action)
        if not command:
            return Response(
                {"status": "error", "message": "Unknown action"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            completed = subprocess.run(
                command,
                cwd=str(settings.BASE_DIR),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Response(
                {"status": "error", "message": "Command timed out after 120s"},
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except Exception as exc:
            return Response(
                {"status": "error", "message": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        output = (completed.stdout or "") + (
            ("\n" + completed.stderr) if completed.stderr else ""
        )
        return Response(
            {
                "status": "ok" if completed.returncode == 0 else "error",
                "exit_code": completed.returncode,
                "command": " ".join(command),
                "output": output.strip(),
            }
        )


from ..models import SharedDesign
from .serializers import SharedDesignSerializer

class SharedDesignViewSet(viewsets.ModelViewSet):
    queryset = SharedDesign.objects.all()
    serializer_class = SharedDesignSerializer
    permission_classes = [permissions.AllowAny]
    
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


from django.http import StreamingHttpResponse, Http404
import requests

def telegram_proxy_view(request, file_id):
    """
    Proxies a Telegram file_id to the frontend securely, avoiding the 1-hour expiration limit.
    """
    token = getattr(settings, 'TELEGRAM_BOT_TOKEN', None)
    if not token:
        raise Http404("Bot token is missing")

    try:
        # First, fetch file info to get the file_path
        info_url = f"https://api.telegram.org/bot{token}/getFile?file_id={file_id}"
        info_res = requests.get(info_url, timeout=10)
        info_res.raise_for_status()
        
        info_data = info_res.json()
        if not info_data.get('ok'):
            raise Http404("Telegram API rejected file_id")
            
        file_path = info_data["result"]["file_path"]
        download_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
        
        # Stream the image file back to the client
        image_res = requests.get(download_url, stream=True, timeout=30)
        image_res.raise_for_status()
        
        response = StreamingHttpResponse(image_res.raw, content_type="image/jpeg")
        response['Cache-Control'] = 'public, max-age=86400' # Cache for 1 day
        return response
        
    except Exception as e:
        logger.error(f"Telegram proxy error: {e}")
        raise Http404("Could not fetch image")

# ── Payment submission (company owner) ───────────────────────
class PaymentViewSet(viewsets.ModelViewSet):
    """
    Company owner submits subscription payments with a screenshot.
    - POST /api/payments/   — upload screenshot + amount + months
    - GET  /api/payments/   — list the current user's company's payments
    - GET  /api/payments/{id}/ — detail (owner only)

    Updates/deletes are disabled — once submitted, only admin can change
    status via the admin endpoints. This prevents owners from tampering
    with a pending payment after admin has already started reviewing.
    """
    serializer_class = PaymentSerializer
    permission_classes = [permissions.AllowAny]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser, parsers.JSONParser]
    http_method_names = ["get", "post", "head", "options"]

    def get_queryset(self):
        tg_user = get_tg_user(self.request)
        if tg_user is None:
            return Payment.objects.none()
        # Owner sees only their own company's payments. Staff sees everything
        # (kept here for debugging — admins usually use /admin/payments/).
        if self.request.user.is_authenticated and self.request.user.is_staff:
            return Payment.objects.select_related("company", "reviewed_by").all()
        return Payment.objects.select_related("company", "reviewed_by").filter(
            company__user=tg_user
        )

    def perform_create(self, serializer):
        tg_user = require_tg_user(self.request)
        company = getattr(tg_user, "company", None)
        if company is None:
            raise ValidationError(
                {"detail": "Avval kompaniya profilini yarating."}
            )

        # Lazy import — notifications uses requests which may not be needed
        # in every code path (e.g. migrations).
        from ..notifications import NotificationService

        payment = serializer.save(company=company)
        
        # Update company status to review
        if company.status == 'pending':
            company.status = 'review'
            company.save(update_fields=['status'])
            
        NotificationService.notify_payment_submitted(payment)


