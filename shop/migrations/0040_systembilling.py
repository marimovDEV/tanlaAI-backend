from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0039_fix_product_is_active_default'),
    ]

    operations = [
        migrations.CreateModel(
            name='SystemBilling',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('server_due_date', models.DateField(blank=True, null=True)),
                ('server_cost', models.IntegerField(default=0, help_text='UZS / oy')),
                ('server_note', models.CharField(blank=True, default='', max_length=255)),
                ('ai_due_date', models.DateField(blank=True, null=True)),
                ('ai_cost_per_request', models.DecimalField(
                    decimal_places=6, default=0.01, help_text='USD / bitta AI so\'rov', max_digits=10
                )),
                ('usd_to_uzs_rate', models.IntegerField(default=12500, help_text='1 USD = X UZS')),
                ('ai_monthly_budget_uzs', models.IntegerField(default=0, help_text='Oylik AI limit (UZS)')),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'System Billing',
            },
        ),
    ]
