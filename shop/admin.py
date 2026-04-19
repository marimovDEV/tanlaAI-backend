from django.db import models
from django.contrib import admin
from django.utils.html import format_html
from .models import Category, Product, Company, TelegramUser, ProductImage

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'icon')


class ProductImageInline(admin.TabularInline):
    model = ProductImage
    extra = 0
    max_num = ProductImage.MAX_IMAGES_PER_PRODUCT
    fields = ('image', 'is_main', 'order')


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ('display_thumbnail', 'name', 'price', 'category', 'is_featured', 'ai_status', 'lead_time_days')
    list_filter = ('category', 'is_featured', 'ai_status')
    search_fields = ('name', 'description')
    readonly_fields = ('display_original', 'display_processed', 'ai_status')
    inlines = [ProductImageInline]

    fieldsets = (
        (None, {
            'fields': ('name', 'description', 'category', 'company', 'owner', 'is_featured')
        }),
        ('Pricing & Dimensions', {
            'fields': ('price', 'price_per_m2', 'height', 'width', 'lead_time_days')
        }),
        ('Visuals (View Only)', {
            'fields': ('display_original', 'display_processed'),
        }),
        ('Image Management', {
            'fields': ('image', 'original_image', 'image_no_bg'),
            'classes': ('collapse',),
        }),
        ('AI Processing', {
            'fields': ('ai_status',),
        }),
        ('Discounts', {
            'fields': ('is_on_sale', 'discount_price', 'sale_end_date'),
            'classes': ('collapse',),
        }),
    )

    def display_thumbnail(self, obj):
        if obj.image:
            return format_html('<img src="{}" style="width: 50px; height: auto; border-radius: 4px;" />', obj.image.url)
        return "-"
    display_thumbnail.short_description = 'Rasm'

    def display_original(self, obj):
        if obj.original_image:
            return format_html('<img src="{}" style="max-width: 300px; height: auto; border: 1px solid #ccc;" />', obj.original_image.url)
        return "Asl rasm yuklanmagan"
    display_original.short_description = 'Asl rasm'

    def display_processed(self, obj):
        if obj.image_no_bg:
            return format_html('<img src="{}" style="max-width: 300px; height: auto; border: 1px solid #4fb; background: #eee;" />', obj.image_no_bg.url)
        return "AI rasm hali tayyor emas"
    display_processed.short_description = 'AI ishlov bergan rasm'

@admin.register(TelegramUser)
class TelegramUserAdmin(admin.ModelAdmin):
    list_display = ('telegram_id', 'first_name', 'username', 'role', 'created_at')
    search_fields = ('telegram_id', 'first_name', 'username')

@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'location', 'created_at')
    search_fields = ('name', 'location', 'user__username')
