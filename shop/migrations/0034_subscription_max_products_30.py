# Raise the baseline product cap from 10 to 30 for new subscriptions.
# Existing Subscription rows are left untouched — admins adjust those per-company.
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("shop", "0033_leadrequest_direct_order"),
    ]

    operations = [
        migrations.AlterField(
            model_name="subscription",
            name="max_products",
            field=models.IntegerField(default=30),
        ),
    ]
