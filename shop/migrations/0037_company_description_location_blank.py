from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('shop', '0036_payment'),
    ]

    operations = [
        migrations.AlterField(
            model_name='company',
            name='description',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AlterField(
            model_name='company',
            name='location',
            field=models.CharField(blank=True, default='', max_length=255),
        ),
    ]
