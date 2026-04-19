# Hand-written migration — adds checkout address fields to LeadRequest:
#   - address_text (free-text)
#   - latitude / longitude (geo-coords)
# Both optional at the DB level; the serializer enforces "at least one" for
# call/measurement lead types.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("shop", "0031_product_lead_time_and_images_and_lead_measurements"),
    ]

    operations = [
        migrations.AddField(
            model_name="leadrequest",
            name="address_text",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="leadrequest",
            name="latitude",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="leadrequest",
            name="longitude",
            field=models.FloatField(blank=True, null=True),
        ),
    ]
