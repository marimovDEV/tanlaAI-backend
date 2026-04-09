import os
import shutil
from django.core.management.base import BaseCommand
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings
from shop.models import Category, Company, Product, TelegramUser, HomeBanner

class Command(BaseCommand):
    help = 'Seeds the database with initial mock data for testing.'

    def handle(self, *args, **kwargs):
        self.stdout.write('Clearing existing data...')
        Product.objects.all().delete()
        Company.objects.all().delete()
        Category.objects.all().delete()
        TelegramUser.objects.all().delete()
        HomeBanner.objects.all().delete()

        # Create Mock User
        user, _ = TelegramUser.objects.get_or_create(
            telegram_id=123456789,
            defaults={
                'first_name': 'Test',
                'last_name': 'User',
                'username': 'testuser',
                'role': 'COMPANY'
            }
        )

        # Create Categories
        self.stdout.write('Kategoriyalar yaratilmoqda...')
        cat_doors = Category.objects.create(name="Premium Eshiklar")
        cat_windows = Category.objects.create(name="Aqlli Oynalar")
        cat_furniture = Category.objects.create(name="Noyob Mebellar")

        # Create Companies
        self.stdout.write('Kompaniyalar yaratilmoqda...')
        comp1 = Company.objects.create(
            user=user,
            name='Luxury Woodworks',
            description="Eng zamonaviy va bejirim eshiklar, mebellar ishlab chiqaruvchi raqamli bo'lim.",
            location='Toshkent, Yunusobod',
            is_active=True
        )
        
        user2, _ = TelegramUser.objects.get_or_create(telegram_id=987654321, defaults={'first_name': 'Modern', 'role': 'COMPANY'})
        comp2 = Company.objects.create(
            user=user2,
            name='Modern Glass & Co.',
            description='Zamonaviy xonadonlar uchun innovatsion oyna yechimlari.',
            location='Samarqand',
            is_active=True
        )

        user3, _ = TelegramUser.objects.get_or_create(telegram_id=112233445, defaults={'first_name': 'Vintage', 'role': 'COMPANY'})
        comp3 = Company.objects.create(
            user=user3,
            name='Vintage Crafts',
            description="Olijanob, mumtoz va sifatli mebellarning ishonchli ta'minotchisi.",
            location='Buxoro',
            is_active=True
        )

        # Create Products
        self.stdout.write('Mahsulotlar yaratilmoqda...')
        # Helper for mock image (we just need a valid image file, but for local we can leave it blank and frontend uses placeholder, or we can use a solid color script)
        
        products_data = [
            # Eshiklar
            {'name': "Viktoriya uslubidagi eman eshik", 'cat': cat_doors, 'comp': comp1, 'price': 1200000, 'is_featured': True, 'sale': False},
            {'name': "Minimalist oq eshik", 'cat': cat_doors, 'comp': comp1, 'price': 850000, 'is_featured': False, 'sale': True, 'discount': 700000},
            {'name': "Mumtoz yong'oq daraxti eshigi", 'cat': cat_doors, 'comp': comp3, 'price': 2500000, 'is_featured': True, 'sale': False},
            # Oynalar
            {'name': "Panoramik aqlli oyna", 'cat': cat_windows, 'comp': comp2, 'price': None, 'm2': 450000, 'is_featured': True, 'sale': False},
            {'name': "Ovoz o'tkazmaydigan qo'sh oyna", 'cat': cat_windows, 'comp': comp2, 'price': None, 'm2': 300000, 'is_featured': False, 'sale': True, 'discount': 250000},
            # Mebellar
            {'name': "Ergonomik ofis stuli", 'cat': cat_furniture, 'comp': comp1, 'price': 1500000, 'is_featured': True, 'sale': False},
            {'name': "Qizil daraxtli ovqatlanish stoli", 'cat': cat_furniture, 'comp': comp3, 'price': 4800000, 'is_featured': True, 'sale': True, 'discount': 4000000},
            {'name': "Kutubxona peshtaxtasi (osiluvchi)", 'cat': cat_furniture, 'comp': comp1, 'price': 600000, 'is_featured': False, 'sale': False},
        ]

        for p in products_data:
            Product.objects.create(
                name=p['name'],
                description=f"Eng yuqori sifatli materiallardan tayyorlangan go'zal va bejirim {p['name'].lower()}.",
                category=p['cat'],
                company=p['comp'],
                owner=p['comp'].user,
                price=p.get('price'),
                price_per_m2=p.get('m2'),
                is_featured=p['is_featured'],
                is_on_sale=p['sale'],
                discount_price=p.get('discount')
            )

        # Create Banners
        self.stdout.write('Bannerlar yaratilmoqda...')
        HomeBanner.objects.create(
            title="Bahorgi Chegirmalar 2026",
            subtitle="Barcha premium eshiklarga 40% gacha chergima!",
            order=1
        )
        HomeBanner.objects.create(
            title="Yangi To'plam",
            subtitle="Zamonaviy oyna yechimlarini kashf eting",
            order=2
        )

        self.stdout.write(self.style.SUCCESS("Ma'lumotlar bazasi muvaffaqiyatli to'ldirildi!"))
