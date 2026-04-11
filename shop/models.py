import os
import io
from PIL import Image
from django.core.files.base import ContentFile
from django.db import models

class Category(models.Model):
    name = models.CharField(max_length=100)
    icon = models.ImageField(upload_to='category_icons/', null=True, blank=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name_plural = "Categories"

class TelegramUser(models.Model):
    ROLE_CHOICES = [
        ('USER', 'User'),
        ('COMPANY', 'Company'),
        ('ADMIN', 'Admin'),
    ]
    telegram_id = models.BigIntegerField(unique=True)
    first_name = models.CharField(max_length=255, null=True, blank=True)
    last_name = models.CharField(max_length=255, null=True, blank=True)
    username = models.CharField(max_length=255, null=True, blank=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='USER')
    photo_url = models.URLField(max_length=1000, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.first_name} ({self.telegram_id})"

class Company(models.Model):
    user = models.OneToOneField(TelegramUser, on_delete=models.CASCADE, related_name='company')
    name = models.CharField(max_length=255)
    description = models.TextField()
    location = models.CharField(max_length=255)
    telegram_link = models.CharField(max_length=255, null=True, blank=True)
    instagram_link = models.CharField(max_length=255, null=True, blank=True)
    logo = models.ImageField(upload_to='company_logos/', null=True, blank=True)
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
    image = models.ImageField(upload_to='products/', null=True, blank=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    height = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True, help_text="Height in cm")
    width = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True, help_text="Width in cm")
    price_per_m2 = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True, help_text="Price for 1 m² (сум/м²)")
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='products', null=True, blank=True)
    owner = models.ForeignKey(TelegramUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='added_products')
    is_featured = models.BooleanField(default=False)
    
    # AI Processing fields
    original_image = models.ImageField(upload_to='products/raw/', null=True, blank=True)
    image_no_bg = models.ImageField(upload_to='products/transparent/', null=True, blank=True)
    ai_status = models.CharField(
        max_length=20, 
        choices=[
            ('none', 'None'),
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('error', 'Error')
        ], 
        default='none'
    )
    ai_error = models.TextField(null=True, blank=True)
    
    # Discount fields
    discount_price = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    is_on_sale = models.BooleanField(default=False)
    sale_end_date = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.name

class AIResult(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('done', 'Done'),
        ('error', 'Error'),
    ]
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE, related_name='ai_results')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='ai_visualizations')
    input_image = models.ImageField(upload_to='ai_inputs/', null=True, blank=True)
    image = models.ImageField(upload_to='ai_results/')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='done')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"AI Result: {self.product.name} for {self.user.first_name}"

class HomeBanner(models.Model):
    title = models.CharField(max_length=200, blank=True)
    subtitle = models.CharField(max_length=500, blank=True)
    image = models.ImageField(upload_to='banners/')
    is_active = models.BooleanField(default=True)
    order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['order', '-created_at']

    def __str__(self):
        return self.title or f"Banner {self.id}"


class Wishlist(models.Model):
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE, related_name='wishlist_items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='wishlisted_by')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'product')
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.first_name} ❤ {self.product.name}"


class LeadRequest(models.Model):
    LEAD_TYPES = [
        ('call', 'Call Request'),
        ('telegram', 'Telegram Message'),
        ('measurement', 'Measurement Request'),
    ]
    user = models.ForeignKey(TelegramUser, on_delete=models.CASCADE, related_name='lead_requests')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='leads')
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='leads', null=True, blank=True)
    lead_type = models.CharField(max_length=20, choices=LEAD_TYPES)
    message = models.TextField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_processed = models.BooleanField(default=False)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.get_lead_type_display()} — {self.product.name}"


class Subscription(models.Model):
    PLANS = [
        ('free', 'Free'),
        ('pro', 'Pro'),
        ('premium', 'Premium'),
    ]
    company = models.OneToOneField(Company, on_delete=models.CASCADE, related_name='subscription')
    plan = models.CharField(max_length=10, choices=PLANS, default='free')
    max_products = models.IntegerField(default=10)
    ai_generations_limit = models.IntegerField(default=50)
    started_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} — {self.get_plan_display()}"


class SystemSetting(models.Model):
    telegram_bot_token = models.CharField(max_length=255, blank=True, default="")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "System Setting"
        verbose_name_plural = "System Settings"

    def __str__(self):
        return "System settings"

    @classmethod
    def get_solo(cls):
        return cls.objects.first() or cls.objects.create()
