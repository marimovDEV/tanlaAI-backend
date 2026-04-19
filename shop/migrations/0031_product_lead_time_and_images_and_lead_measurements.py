# Hand-written migration — adds:
#   - Product.lead_time_days
#   - ProductImage (5 images per product, 1 main = cutout)
#   - LeadRequest.width_cm / height_cm / calculated_price

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("shop", "0030_company_latitude_company_longitude_company_phone_and_more"),
    ]

    operations = [
        # --- Product: lead_time_days ---
        migrations.AddField(
            model_name="product",
            name="lead_time_days",
            field=models.PositiveIntegerField(
                default=3,
                help_text="Necha kunda tayyor bo'ladi (kun)",
            ),
        ),

        # --- LeadRequest: structured measurement fields ---
        migrations.AddField(
            model_name="leadrequest",
            name="width_cm",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="leadrequest",
            name="height_cm",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="leadrequest",
            name="calculated_price",
            field=models.DecimalField(
                blank=True,
                decimal_places=2,
                help_text="Auto-calculated: (w*h/10000) * price_per_m2",
                max_digits=12,
                null=True,
            ),
        ),

        # --- New ProductImage model (gallery with 1 cutout + up to 4 showcase) ---
        migrations.CreateModel(
            name="ProductImage",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("image", models.ImageField(upload_to="products/gallery/")),
                (
                    "is_main",
                    models.BooleanField(
                        default=False,
                        help_text=(
                            "Asosiy (foni olingan) rasm. Mahsulot bo'yicha "
                            "faqat 1 ta True bo'ladi."
                        ),
                    ),
                ),
                ("order", models.PositiveSmallIntegerField(default=0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "product",
                    models.ForeignKey(
                        on_delete=models.deletion.CASCADE,
                        related_name="images",
                        to="shop.product",
                    ),
                ),
            ],
            options={
                "verbose_name": "Product Image",
                "verbose_name_plural": "Product Images",
                "ordering": ["-is_main", "order", "id"],
            },
        ),
    ]
