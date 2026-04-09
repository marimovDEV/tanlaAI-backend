from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from shop.views import admin_dashboard_home

urlpatterns = [
    path('admin/', admin_dashboard_home, name='admin_dashboard_home_redirect'),
    path('', include('shop.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
