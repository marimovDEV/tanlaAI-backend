import io
import os

from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.db import models, transaction
from PIL import Image


class Category(models.Model):
    name = models.CharField(max_length=100)
    icon = models.ImageField(upload_to="category_icons/", null=True, blank=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name_plural = "Categories"


class TelegramUser(models.Model):
    ROLE_CHOICES = [
        ("USER", "User"),
        ("COMPANY", "Company"),
        ("ADMIN", "Admin"),
    ]
    telegram_id = models.BigIntegerField(unique=True)
    first_name = models.CharField(max_length=255, null=True, blank=True)
    last_name = models.CharField(max_length=255, null=True, blank=True)
    username = models.CharField(max_length=255, null=True, blank=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default="USER")
    photo_url = models.URLField(max_length=1000, null=True, blank=True)
    phone = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.first_name} ({self.telegram_id})"


class Company(models.Model):
    user = models.OneToOneField(
        TelegramUser, on_delete=models.CASCADE, related_name="company"
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    phone = models.CharField(max_length=20, blank=True, default='')
    location = models.CharField(max_length=255, blank=True, default='')
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    telegram_link = models.CharField(max_length=255, blank=True, default='')
    instagram_link = models.CharField(max_length=255, blank=True, default='')
    youtube_link = models.CharField(max_length=255, blank=True, default='')
    logo = models.ImageField(upload_to="company_logos/", null=True, blank=True)
    is_active = models.BooleanField(default=True)
    subscription_deadline = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def is_currently_active(self):
        from django.utils import timezone

        if not self.is_active:
            return False
        if self.subscription_deadline and self.subscription_deadline < timezone.now():
            return False
        return True

    def __str__(self):
        return self.name


class Product(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    price = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    image = models.ImageField(upload_to="products/", null=True, blank=True)
    category = models.ForeignKey(
        Category, on_delete=models.CASCADE, related_name="products"
    )
    height = models.DecimalField(
        max_digits=10, decimal_places=2, null=True, blank=True, help_text="Height in cm"
    )
    width = models.DecimalField(
        max_digits=10, decimal_places=2, null=True, blank=True, help_text="Width in cm"
    )
    price_per_m2 = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Price for 1 m² (сум/м²)",
    )
    company = models.ForeignKey(
        Company,
        on_delete=models.CASCADE,
        related_name="products",
        null=True,
        blank=True,
    )
    owner = models.ForeignKey(
        TelegramUser,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="added_products",
    )
    is_featured = models.BooleanField(default=False)
    # Owner can pause a listing without deleting it. The cron also flips this
    # off for every product when the company's subscription expires.
    # Public listings hide is_active=False; the owner's own dashboard ("my")
    # still shows them so they know what's paused.
    is_active = models.BooleanField(default=True)

    # AI Processing fields
    original_image = models.ImageField(upload_to="products/raw/", null=True, blank=True)
    image_no_bg = models.ImageField(
        upload_to="products/transparent/", null=True, blank=True
    )
    ai_status = models.CharField(
        max_length=20,
        choices=[
            ("none", "None"),
            ("processing", "Processing"),
            ("completed", "Completed"),
            ("error", "Error"),
        ],
        default="none",
    )
    ai_error = models.TextField(null=True, blank=True)

    # Discount fields
    discount_price = models.DecimalField(
        max_digits=15, decimal_places=2, null=True, blank=True
    )
    is_on_sale = models.BooleanField(default=False)
    sale_end_date = models.DateTimeField(null=True, blank=True)

    # Lead time (ready-by duration, in days)
    lead_time_days = models.PositiveIntegerField(
        default=3,
        help_text="Necha kunda tayyor bo'ladi (kun)",
    )

    def __str__(self):
        return self.name


class ProductImage(models.Model):
    """
    Additional images for a Product.
    A product can have up to 5 images:
      - Exactly 1 with `is_main=True` (the cutout / background-removed image)
      - Up to 4 additional "showcase" images (interior / real previews)
    The legacy `Product.image` / `image_no_bg` fields are kept untouched for
    backward compatibility with the existing AI pipeline.
    """

    MAX_IMAGES_PER_PRODUCT = 5

    product = models.ForeignKey(
        Product, on_delete=models.CASCADE, related_name="images"
    )
    image = models.ImageField(upload_to="products/gallery/")
    is_main = models.BooleanField(
        default=False,
        help_text="Asosiy (foni olingan) rasm. Mahsulot bo'yicha faqat 1 ta True bo'ladi.",
    )
    order = models.PositiveSmallIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-is_main", "order", "id"]
        verbose_name = "Product Image"
        verbose_name_plural = "Product Images"

    def __str__(self):
        tag = "main" if self.is_main else "showcase"
        return f"{self.product.name} [{tag} #{self.order}]"

    def save(self, *args, **kwargs):
        # Production-safe invariants:
        #   a) Hard cap: never more than MAX_IMAGES_PER_PRODUCT per product.
        #   b) Single main: exactly one is_main=True row per product.
        # Both must be atomic so concurrent POSTs can't bypass them.
        with transaction.atomic():
            if not self.pk and self.product_id:
                # Lock the product's existing images for the duration of this txn.
                existing = list(
                    ProductImage.objects.select_for_update()
                    .filter(product_id=self.product_id)
                    .values_list("id", flat=True)
                )
                if len(existing) >= self.MAX_IMAGES_PER_PRODUCT:
                    raise ValidationError(
                        f"Max {self.MAX_IMAGES_PER_PRODUCT} ta rasm ruxsat etilgan"
                    )

            if self.is_main and self.product_id:
                # Demote any other main rows atomically.
                ProductImage.objects.select_for_update().filter(
                    product_id=self.product_id, is_main=True
                ).exclude(pk=self.pk).update(is_main=False)

            super().save(*args, **kwargs)


class AIResult(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("done", "Done"),
        ("error", "Error"),
    ]
    user = models.ForeignKey(
        TelegramUser, on_delete=models.CASCADE, related_name="ai_results"
    )
    product = models.ForeignKey(
        Product, on_delete=models.CASCADE, related_name="ai_visualizations"
    )
    input_image = models.ImageField(upload_to="ai_inputs/", null=True, blank=True)
    image = models.ImageField(upload_to="ai_results/")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="done")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"AI Result: {self.product.name} for {self.user.first_name}"


class HomeBanner(models.Model):
    title = models.CharField(max_length=200, blank=True)
    subtitle = models.CharField(max_length=500, blank=True)
    image = models.ImageField(upload_to="banners/")
    is_active = models.BooleanField(default=True)
    order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["order", "-created_at"]

    def __str__(self):
        return self.title or f"Banner {self.id}"


class Wishlist(models.Model):
    user = models.ForeignKey(
        TelegramUser, on_delete=models.CASCADE, related_name="wishlist_items"
    )
    product = models.ForeignKey(
        Product, on_delete=models.CASCADE, related_name="wishlisted_by"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "product")
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.first_name} ❤ {self.product.name}"


class LeadRequest(models.Model):
    LEAD_TYPES = [
        ("call", "Call Request"),
        ("telegram", "Telegram Message"),
        ("measurement", "Measurement Request"),
        ("visualize", "AI Visualization"),
        # Direct checkout — customer orders the product without going through
        # the AI visualization flow. Requires phone + address (lat/lng OR text).
        ("direct", "Direct Order"),
    ]
    STATUS_CHOICES = [
        ("new", "🆕 Yangi"),
        ("contacted", "📞 Bog'lanildi"),
        ("active", "⚡️ Jarayonda"),
        ("converted", "✅ Sotildi"),
        ("rejected", "❌ Rad etildi"),
        ("closed", "🔒 Yakunlandi"),
    ]
    user = models.ForeignKey(
        TelegramUser, on_delete=models.CASCADE, related_name="lead_requests"
    )
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="leads")
    company = models.ForeignKey(
        Company, on_delete=models.CASCADE, related_name="leads", null=True, blank=True
    )
    ai_result = models.ForeignKey(
        "AIResult",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="linked_leads",
    )
    lead_type = models.CharField(max_length=20, choices=LEAD_TYPES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="new")
    message = models.TextField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    price_info = models.CharField(
        max_length=100, blank=True, help_text="Calculated price or dimensions info"
    )
    # Structured measurement fields (filled when user orders via o'lchash flow)
    width_cm = models.PositiveIntegerField(null=True, blank=True)
    height_cm = models.PositiveIntegerField(null=True, blank=True)
    calculated_price = models.DecimalField(
        max_digits=12, decimal_places=2, null=True, blank=True,
        help_text="Auto-calculated: (w*h/10000) * price_per_m2"
    )
    # Delivery / visit address — either free-text or geo-coords (or both).
    address_text = models.TextField(blank=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    quantity = models.PositiveIntegerField(default=1)
    total_price = models.DecimalField(
        max_digits=14, decimal_places=2, null=True, blank=True,
        help_text="quantity × price, computed on frontend and stored"
    )
    source = models.CharField(max_length=50, blank=True)
    shared_id = models.UUIDField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_processed = models.BooleanField(default=False)  # Keep for compatibility
    is_paid = models.BooleanField(default=False)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.first_name} — {self.product.name} ({self.get_status_display()})"


class Subscription(models.Model):
    PLANS = [
        ("free", "Free"),
        ("pro", "Pro"),
        ("premium", "Premium"),
    ]
    company = models.OneToOneField(
        Company, on_delete=models.CASCADE, related_name="subscription"
    )
    plan = models.CharField(max_length=10, choices=PLANS, default="free")
    # 30 is the baseline marketplace cap. Paid tiers can bump this via admin.
    max_products = models.IntegerField(default=30)
    ai_generations_limit = models.IntegerField(default=50)
    started_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} — {self.get_plan_display()}"


class Payment(models.Model):
    """
    Subscription payment submission.

    Flow:
      1. Company owner pays via bank/card, uploads a screenshot → status="pending".
      2. Admin reviews in /adminka/payments and clicks Approve or Reject.
      3. On approve: `Company.subscription_deadline` is extended by `months`
         from the LATER of (now, current deadline) — so early payments stack
         instead of being wasted. Paused products belonging to the company
         are reactivated in the same transaction.
      4. On reject: status="rejected" + rejection_reason is filled. The owner
         is notified via Telegram.

    We deliberately keep this model separate from `Subscription` rather than
    mutating Subscription in place, because we want an audit trail of every
    payment (amount, screenshot, who approved, when).
    """
    STATUS_CHOICES = [
        ("pending", "Pending Review"),
        ("approved", "Approved"),
        ("rejected", "Rejected"),
    ]

    company = models.ForeignKey(
        Company, on_delete=models.CASCADE, related_name="payments"
    )
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    months = models.PositiveSmallIntegerField(
        default=1,
        help_text="How many months to extend the subscription on approval.",
    )
    screenshot = models.ImageField(upload_to="payments/")
    note = models.TextField(blank=True, help_text="Owner's note (bank, reference, etc.)")

    status = models.CharField(
        max_length=10, choices=STATUS_CHOICES, default="pending"
    )
    rejection_reason = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    reviewed_by = models.ForeignKey(
        TelegramUser,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="reviewed_payments",
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.company.name} — {self.amount} ({self.get_status_display()})"


class SystemSettings(models.Model):
    # --- General Settings ---
    platform_name = models.CharField(max_length=100, default="Tanla")
    default_language = models.CharField(max_length=10, default="uz")
    timezone = models.CharField(max_length=50, default="Asia/Tashkent")
    currency = models.CharField(max_length=20, default="UZS")

    # --- AI Settings ---
    ai_provider = models.CharField(
        max_length=50,
        choices=[
            ("gemini_direct", "Gemini Direct (Tavsiya etilgan)"),
            ("gemini", "Gemini Imagen"),
            ("openai", "OpenAI DALL-E"),
            ("hybrid", "Hybrid (OpenCV)"),
        ],
        default="gemini_direct",
    )
    max_results_per_user = models.IntegerField(default=20)
    image_quality = models.CharField(
        max_length=20,
        choices=[("low", "Low"), ("medium", "Medium"), ("high", "High")],
        default="high",
    )
    enable_bg_removal = models.BooleanField(default=True)

    # --- Image Processing ---
    bg_removal_mode = models.CharField(
        max_length=50,
        choices=[("ai", "AI-based"), ("color", "Color-based"), ("hybrid", "Hybrid")],
        default="ai",
    )
    max_image_size = models.IntegerField(default=2048, help_text="Pixels")
    auto_resize = models.BooleanField(default=True)
    keep_aspect_ratio = models.BooleanField(default=True)

    # --- Visualization Settings ---
    default_door_height_ratio = models.FloatField(default=0.65, help_text="0.0 to 1.0")
    placement_mode = models.CharField(
        max_length=50,
        choices=[("center", "Center"), ("auto", "Auto-detect"), ("manual", "Manual")],
        default="auto",
    )
    allow_window_placement = models.BooleanField(default=False)
    snap_to_wall = models.BooleanField(default=True)

    # --- CRM Settings ---
    auto_create_lead = models.BooleanField(default=True)
    notify_admin = models.BooleanField(default=True)
    default_lead_status = models.CharField(max_length=50, default="new")

    # --- Admin Settings ---
    items_per_page = models.IntegerField(default=20)
    enable_infinite_scroll = models.BooleanField(default=True)
    show_debug_logs = models.BooleanField(default=False)

    # --- DevOps & Updates ---
    enable_deploy_actions = models.BooleanField(default=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "System Setting"
        verbose_name_plural = "System Settings"

    def __str__(self):
        return f"{self.platform_name} Settings"

    @classmethod
    def get_solo(cls):
        obj, created = cls.objects.get_or_create(id=1)
        return obj


class AITest(models.Model):
    door = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="ai_tests")
    room_image = models.ImageField(upload_to="ai_tests/rooms/")
    prompt = models.TextField(blank=True, null=True)
    result_image = models.ImageField(
        upload_to="ai_tests/results/", blank=True, null=True
    )
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "AI Test"
        verbose_name_plural = "AI Tests"

    def __str__(self):
        return f"Test #{self.id}: {self.door.name}"


import uuid

class SharedDesign(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to="shared_designs/")
    original_image = models.ImageField(upload_to="shared_designs/originals/", null=True, blank=True)
    product = models.ForeignKey(Product, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Shared Design"
        verbose_name_plural = "Shared Designs"

    def __str__(self):
        return f"Shared Design: {str(self.id)[:8]}..."
