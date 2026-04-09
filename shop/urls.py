from django.contrib.auth.views import LogoutView
from django.urls import include, path, re_path

from .views import (
    AdminLoginView,
    admin_banner_create,
    admin_banner_delete,
    admin_banner_edit,
    admin_banner_list,
    admin_category_create,
    admin_category_delete,
    admin_category_edit,
    admin_category_list,
    admin_company_delete,
    admin_company_edit,
    admin_company_list,
    admin_company_toggle_active,
    admin_company_update_deadline,
    admin_dashboard_home,
    admin_product_create,
    admin_product_delete,
    admin_product_edit,
    admin_product_list,
    admin_promotion_list,
    admin_user_list,
    admin_user_toggle_role,
    auth_login,
)
from .views.spa import spa_entry_view


urlpatterns = [
    path('login/', AdminLoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(next_page='home'), name='logout'),
    path('auth/login/', auth_login, name='auth_login'),

    path('admin-dashboard/', admin_dashboard_home, name='admin_dashboard'),
    path('admin-dashboard/products/', admin_product_list, name='admin_product_list'),
    path('admin-dashboard/products/create/', admin_product_create, name='admin_product_create'),
    path('admin-dashboard/products/edit/<int:pk>/', admin_product_edit, name='admin_product_edit'),
    path('admin-dashboard/products/delete/<int:pk>/', admin_product_delete, name='admin_product_delete'),

    path('admin-dashboard/categories/', admin_category_list, name='admin_category_list'),
    path('admin-dashboard/categories/create/', admin_category_create, name='admin_category_create'),
    path('admin-dashboard/categories/edit/<int:pk>/', admin_category_edit, name='admin_category_edit'),
    path('admin-dashboard/categories/delete/<int:pk>/', admin_category_delete, name='admin_category_delete'),

    path('admin-dashboard/users/', admin_user_list, name='admin_user_list'),
    path('admin-dashboard/users/toggle-role/<int:pk>/', admin_user_toggle_role, name='admin_user_toggle_role'),

    path('admin-dashboard/companies/', admin_company_list, name='admin_company_list'),
    path('admin-dashboard/companies/<int:pk>/edit/', admin_company_edit, name='admin_company_edit'),
    path('admin-dashboard/companies/<int:pk>/delete/', admin_company_delete, name='admin_company_delete'),
    path('admin-dashboard/companies/toggle-active/<int:pk>/', admin_company_toggle_active, name='admin_company_toggle_active'),
    path('admin-dashboard/companies/update-deadline/<int:pk>/', admin_company_update_deadline, name='admin_company_update_deadline'),

    path('admin-dashboard/promotions/', admin_promotion_list, name='admin_promotion_list'),

    path('admin-dashboard/banners/', admin_banner_list, name='admin_banner_list'),
    path('admin-dashboard/banners/create/', admin_banner_create, name='admin_banner_create'),
    path('admin-dashboard/banners/edit/<int:pk>/', admin_banner_edit, name='admin_banner_edit'),
    path('admin-dashboard/banners/delete/<int:pk>/', admin_banner_delete, name='admin_banner_delete'),

    path('api/v1/', include('shop.api.urls')),

    path('', spa_entry_view, name='home'),
    re_path(r'^(?!api/v1/|auth/login/|login/|logout/|admin-dashboard/|admin/|media/|static/).+$', spa_entry_view, name='spa_entry'),
]
