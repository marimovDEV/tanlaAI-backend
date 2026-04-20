# Add owner/cron-controlled is_active flag to Product.
# Default True so every existing row stays visible after migration.
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("shop", "0034_subscription_max_products_30"),
    ]

    operations = [
        migrations.AddField(
            model_name="product",
            name="is_active",
            field=models.BooleanField(default=True),
        ),
    ]
