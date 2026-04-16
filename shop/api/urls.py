from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    TelegramAuthView, CategoryViewSet, ProductViewSet, 
    CompanyViewSet, BannerViewSet, WishlistViewSet, 
    LeadRequestViewSet, AIResultViewSet, SharedDesignViewSet,
    AdminLoginApiView, AdminLogoutApiView, AdminMeApiView,
    AdminSystemSettingsApiView, AdminRunActionApiView,
)

from .admin_api import (
    AdminDashboardApiView,
    AdminProductViewSet,
    AdminCategoryViewSet,
    AdminUserViewSet,
    AdminCompanyViewSet,
    AdminPromotionViewSet,
    AdminBannerViewSet,
    AdminLeadViewSet,
    AdminAIResultViewSet,
    AdminAITestViewSet,
)

# Public API router
router = DefaultRouter()
router.register(r'categories', CategoryViewSet)
router.register(r'products', ProductViewSet)
router.register(r'companies', CompanyViewSet)
router.register(r'banners', BannerViewSet)
router.register(r'wishlist', WishlistViewSet, basename='wishlist')
router.register(r'leads', LeadRequestViewSet, basename='leads')
router.register(r'ai-results', AIResultViewSet, basename='ai-results')
router.register(r'shared-designs', SharedDesignViewSet, basename='shared-designs')

# Admin panel API router
admin_router = DefaultRouter()
admin_router.register(r'products', AdminProductViewSet, basename='admin-products')
admin_router.register(r'categories', AdminCategoryViewSet, basename='admin-categories')
admin_router.register(r'users', AdminUserViewSet, basename='admin-users')
admin_router.register(r'companies', AdminCompanyViewSet, basename='admin-companies')
admin_router.register(r'promotions', AdminPromotionViewSet, basename='admin-promotions')
admin_router.register(r'banners', AdminBannerViewSet, basename='admin-banners')
admin_router.register(r'leads', AdminLeadViewSet, basename='admin-leads')
admin_router.register(r'ai-results', AdminAIResultViewSet, basename='admin-ai-results')
admin_router.register(r'ai-tests', AdminAITestViewSet, basename='admin-ai-tests')

urlpatterns = [
    path('auth/telegram/', TelegramAuthView.as_view(), name='api_tg_auth'),
    path('admin/login/', AdminLoginApiView.as_view(), name='api_admin_login'),
    path('admin/logout/', AdminLogoutApiView.as_view(), name='api_admin_logout'),
    path('admin/me/', AdminMeApiView.as_view(), name='api_admin_me'),
    path('admin/system-settings/', AdminSystemSettingsApiView.as_view(), name='api_admin_system_settings'),
    path('admin/run-action/', AdminRunActionApiView.as_view(), name='api_admin_run_action'),
    path('admin/dashboard/', AdminDashboardApiView.as_view(), name='api_admin_dashboard'),
    path('admin/', include(admin_router.urls)),
    path('', include(router.urls)),
]
