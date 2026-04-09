from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    TelegramAuthView, CategoryViewSet, ProductViewSet, 
    CompanyViewSet, BannerViewSet, WishlistViewSet, 
    LeadRequestViewSet, AIResultViewSet,
    AdminLoginApiView, AdminLogoutApiView, AdminMeApiView,
    AdminSystemSettingsApiView, AdminRunActionApiView,
)

router = DefaultRouter()
router.register(r'categories', CategoryViewSet)
router.register(r'products', ProductViewSet)
router.register(r'companies', CompanyViewSet)
router.register(r'banners', BannerViewSet)
router.register(r'wishlist', WishlistViewSet, basename='wishlist')
router.register(r'leads', LeadRequestViewSet, basename='leads')
router.register(r'ai-results', AIResultViewSet, basename='ai-results')

urlpatterns = [
    path('auth/telegram/', TelegramAuthView.as_view(), name='api_tg_auth'),
    path('admin/login/', AdminLoginApiView.as_view(), name='api_admin_login'),
    path('admin/logout/', AdminLogoutApiView.as_view(), name='api_admin_logout'),
    path('admin/me/', AdminMeApiView.as_view(), name='api_admin_me'),
    path('admin/system-settings/', AdminSystemSettingsApiView.as_view(), name='api_admin_system_settings'),
    path('admin/run-action/', AdminRunActionApiView.as_view(), name='api_admin_run_action'),
    path('', include(router.urls)),
]
