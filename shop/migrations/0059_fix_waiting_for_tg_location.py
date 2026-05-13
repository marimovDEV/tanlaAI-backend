from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [('shop', '0058_product_has_frame_crown')]
    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],  # column already exists in DB
            state_operations=[
                migrations.AddField(
                    model_name='leadrequest',
                    name='waiting_for_tg_location',
                    field=models.BooleanField(default=False),
                ),
            ],
        ),
    ]
