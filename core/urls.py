from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

from django.views.static import serve
import re

urlpatterns = [
    path('admin/', RedirectView.as_view(url='/adminka/login', permanent=False), name='admin_dashboard_home_redirect'),
    path('', include('shop.urls')),
]

# Serve media files in all environments (essential for VPS testing where DEBUG=False)
urlpatterns += [
    path('media/<path:path>', serve, {'document_root': settings.MEDIA_ROOT}),
]

