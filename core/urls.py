from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', RedirectView.as_view(url='/adminka/login', permanent=False), name='admin_dashboard_home_redirect'),
    path('', include('shop.urls')),
]

# Serve media files in all environments as a fallback
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
