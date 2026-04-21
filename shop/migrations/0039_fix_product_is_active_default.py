"""
Fix: set is_active=True for all products that are False due to the BooleanField bug
(FormData without is_active field was interpreted as False by DRF).
"""
from django.db import migrations


def activate_all_products(apps, schema_editor):
    Product = apps.get_model('shop', 'Product')
    updated = Product.objects.filter(is_active=False).update(is_active=True)
    print(f"  Activated {updated} product(s) that were incorrectly inactive.")


class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0038_leadrequest_quantity_total_price'),
    ]

    operations = [
        migrations.RunPython(activate_all_products, migrations.RunPython.noop),
    ]
