# Payment model — owners submit subscription-payment screenshots; admin reviews.
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("shop", "0035_product_is_active"),
    ]

    operations = [
        migrations.CreateModel(
            name="Payment",
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
                ("amount", models.DecimalField(decimal_places=2, max_digits=12)),
                (
                    "months",
                    models.PositiveSmallIntegerField(
                        default=1,
                        help_text="How many months to extend the subscription on approval.",
                    ),
                ),
                ("screenshot", models.ImageField(upload_to="payments/")),
                (
                    "note",
                    models.TextField(
                        blank=True,
                        help_text="Owner's note (bank, reference, etc.)",
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending Review"),
                            ("approved", "Approved"),
                            ("rejected", "Rejected"),
                        ],
                        default="pending",
                        max_length=10,
                    ),
                ),
                ("rejection_reason", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("reviewed_at", models.DateTimeField(blank=True, null=True)),
                (
                    "company",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="payments",
                        to="shop.company",
                    ),
                ),
                (
                    "reviewed_by",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="reviewed_payments",
                        to="shop.telegramuser",
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
