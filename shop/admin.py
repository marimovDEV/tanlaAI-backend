from django.db import models
from django.contrib import admin
from .models import Category, Product, Company, TelegramUser

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'icon')

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ('name', 'price', 'category', 'is_featured')
    list_filter = ('category', 'is_featured')
    search_fields = ('name', 'description')
    readonly_fields = ('ai_status', 'original_image', 'image_no_bg')
    
    fieldsets = (
        (None, {
            'fields': ('name', 'description', 'category', 'company', 'owner', 'is_featured')
        }),
        ('Pricing & Dimensions', {
            'fields': ('price', 'price_per_m2', 'height', 'width')
        }),
        ('Images', {
            'fields': ('image', 'original_image', 'image_no_bg')
        }),
        ('AI Processing', {
            'fields': ('ai_status',),
            'classes': ('collapse',),
        }),
        ('Discounts', {
            'fields': ('is_on_sale', 'discount_price', 'sale_end_date'),
            'classes': ('collapse',),
        }),
    )

@admin.register(TelegramUser)
class TelegramUserAdmin(admin.ModelAdmin):
    list_display = ('telegram_id', 'first_name', 'username', 'role', 'created_at')
    search_fields = ('telegram_id', 'first_name', 'username')

@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'location', 'created_at')
    search_fields = ('name', 'location', 'user__username')
