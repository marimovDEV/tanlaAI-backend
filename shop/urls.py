from django.urls import include, path, re_path
from django.views.generic import RedirectView

from .views import (
    auth_login,
)
from .views.spa import spa_entry_view


urlpatterns = [
    # Legacy server-rendered admin routes are redirected to React adminka.
    path('login/', RedirectView.as_view(url='/adminka/login', permanent=False), name='login'),
    path('logout/', RedirectView.as_view(url='/adminka/login', permanent=False), name='logout'),
    path('auth/login/', auth_login, name='auth_login'),

    path('admin-dashboard/', RedirectView.as_view(url='/adminka/system', permanent=False), name='admin_dashboard'),
    path('admin-dashboard/<path:path>', RedirectView.as_view(url='/adminka/system', permanent=False)),

    path('api/v1/', include('shop.api.urls')),

    path('', spa_entry_view, name='home'),
    re_path(r'^(?!api/v1/|auth/login/|login/|logout/|admin-dashboard/|admin/|media/|static/).+$', spa_entry_view, name='spa_entry'),
]
