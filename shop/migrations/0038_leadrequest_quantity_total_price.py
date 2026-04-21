from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0037_company_description_location_blank'),
    ]

    operations = [
        migrations.AddField(
            model_name='leadrequest',
            name='quantity',
            field=models.PositiveIntegerField(default=1),
        ),
        migrations.AddField(
            model_name='leadrequest',
            name='total_price',
            field=models.DecimalField(
                blank=True, decimal_places=2, max_digits=14, null=True,
                help_text='quantity × price, computed on frontend and stored'
            ),
        ),
    ]
