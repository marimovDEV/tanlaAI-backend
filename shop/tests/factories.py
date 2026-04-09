from itertools import count

from shop.models import Category, Company, Product, TelegramUser


_telegram_ids = count(1000)
_category_names = count(1)
_company_names = count(1)
_product_names = count(1)


def TelegramUserFactory(**kwargs):
    sequence = next(_telegram_ids)
    defaults = {
        'telegram_id': sequence,
        'first_name': f'Test{sequence}',
        'username': f'user{sequence}',
    }
    defaults.update(kwargs)
    return TelegramUser.objects.create(**defaults)


def CategoryFactory(**kwargs):
    sequence = next(_category_names)
    defaults = {
        'name': f'Doors {sequence}',
    }
    defaults.update(kwargs)
    return Category.objects.create(**defaults)


def CompanyFactory(**kwargs):
    defaults = {
        'user': kwargs.pop('user', TelegramUserFactory()),
        'name': f'Company {next(_company_names)}',
        'description': 'Test company description',
        'location': 'Tashkent',
        'is_active': True,
    }
    defaults.update(kwargs)
    return Company.objects.create(**defaults)


def ProductFactory(**kwargs):
    company = kwargs.pop('company', CompanyFactory())
    defaults = {
        'owner': kwargs.pop('owner', company.user),
        'company': company,
        'category': kwargs.pop('category', CategoryFactory()),
        'name': f'Product {next(_product_names)}',
        'description': 'Test product description',
        'price': 1000.0,
    }
    defaults.update(kwargs)
    return Product.objects.create(**defaults)
