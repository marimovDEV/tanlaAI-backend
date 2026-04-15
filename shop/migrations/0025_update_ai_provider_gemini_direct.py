# Generated manually on 2025-01-01

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("shop", "0024_aitest"),
    ]

    operations = [
        migrations.AlterField(
            model_name="systemsettings",
            name="ai_provider",
            field=models.CharField(
                choices=[
                    ("gemini_direct", "Gemini Direct (Tavsiya etilgan)"),
                    ("gemini", "Gemini Imagen"),
                    ("openai", "OpenAI DALL-E"),
                    ("hybrid", "Hybrid (OpenCV)"),
                ],
                default="gemini_direct",
                max_length=50,
            ),
        ),
    ]
