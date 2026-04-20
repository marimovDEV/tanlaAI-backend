# Generated manually for direct-order lead type.
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("shop", "0032_leadrequest_address_fields"),
    ]

    operations = [
        migrations.AlterField(
            model_name="leadrequest",
            name="lead_type",
            field=models.CharField(
                choices=[
                    ("call", "Call Request"),
                    ("telegram", "Telegram Message"),
                    ("measurement", "Measurement Request"),
                    ("visualize", "AI Visualization"),
                    ("direct", "Direct Order"),
                ],
                max_length=20,
            ),
        ),
    ]
