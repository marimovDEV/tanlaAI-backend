from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import user_passes_test
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q, Count
from django.utils import timezone
from .auth import is_staff
from shop.models import Product, Category, TelegramUser, Company, HomeBanner
from shop.forms import ProductForm, CategoryForm, CompanyForm, HomeBannerForm


@user_passes_test(is_staff)
def admin_dashboard_home(request):
    on_sale_products = Product.objects.filter(is_on_sale=True).order_by('-id')
    context = {
        'product_count': Product.objects.count(),
        'company_count': Company.objects.count(),
        'category_count': Category.objects.count(),
        'user_count': TelegramUser.objects.count(),
        'banner_count': HomeBanner.objects.count(),
        'on_sale_products': on_sale_products,
    }
    return render(request, 'admin/dashboard.html', context)


@user_passes_test(is_staff)
def admin_product_list(request):
    query = request.GET.get('q')
    if query:
        products = Product.objects.filter(
            Q(name__icontains=query) | Q(description__icontains=query)
        ).distinct().order_by('-id')
    else:
        products = Product.objects.all().order_by('-id')
    
    if request.headers.get('HX-Request'):
        return render(request, 'admin/partials/product_table_body.html', {'products': products})
        
@user_passes_test(is_staff)
def admin_product_create(request):
    form = ProductForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        form.save()
        return redirect('admin_product_list')
    return render(request, 'admin/product_form.html', {'form': form})


@user_passes_test(is_staff)
def admin_product_edit(request, pk):
    product = get_object_or_404(Product, pk=pk)
    form = ProductForm(request.POST or None, request.FILES or None, instance=product)
    if request.method == 'POST' and form.is_valid():
        form.save()
        return redirect('admin_product_list')
    return render(request, 'admin/product_form.html', {'form': form})


@user_passes_test(is_staff)
def admin_product_delete(request, pk):
    product = get_object_or_404(Product, pk=pk)
    if request.method == 'POST':
        product.delete()
        return redirect('admin_product_list')
    return render(request, 'admin/product_confirm_delete.html', {'product': product})


@user_passes_test(is_staff)
def admin_category_list(request):
    categories = Category.objects.all().order_by('-id')
    return render(request, 'admin/category_list.html', {'categories': categories})


@user_passes_test(is_staff)
def admin_category_create(request):
    form = CategoryForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        form.save()
        return redirect('admin_category_list')
    return render(request, 'admin/category_form.html', {'form': form})


@user_passes_test(is_staff)
def admin_category_edit(request, pk):
    category = get_object_or_404(Category, pk=pk)
    form = CategoryForm(request.POST or None, request.FILES or None, instance=category)
    if request.method == 'POST' and form.is_valid():
        form.save()
        return redirect('admin_category_list')
    return render(request, 'admin/category_form.html', {'form': form})


@user_passes_test(is_staff)
def admin_category_delete(request, pk):
    category = get_object_or_404(Category, pk=pk)
    if request.method == 'POST':
        category.delete()
        return redirect('admin_category_list')
    return render(request, 'admin/category_confirm_delete.html', {'category': category})


@user_passes_test(is_staff)
def admin_user_list(request):
    query = request.GET.get('q')
    if query:
        users = TelegramUser.objects.filter(
            Q(first_name__icontains=query) | 
            Q(last_name__icontains=query) | 
            Q(username__icontains=query)
        ).order_by('-created_at')
    else:
        users = TelegramUser.objects.all().order_by('-created_at')
    
    if request.headers.get('HX-Request'):
        return render(request, 'admin/partials/user_table_body.html', {'users': users})
        
    return render(request, 'admin/user_list.html', {'users': users, 'query': query})


@user_passes_test(is_staff)
def admin_user_toggle_role(request, pk):
    user = get_object_or_404(TelegramUser, pk=pk)
    if user.role == 'USER':
        user.role = 'COMPANY'
    else:
        user.role = 'USER'
    user.save()
    return redirect('admin_user_list')


@user_passes_test(is_staff)
def admin_promotion_list(request):
    products = Product.objects.filter(is_on_sale=True).order_by('-id')
    return render(request, 'admin/promotion_list.html', {'products': products})


@user_passes_test(is_staff)
def admin_company_list(request):
    query = request.GET.get('q')
    companies = Company.objects.annotate(product_count=Count('products')).order_by('-created_at')
    
    if query:
        companies = companies.filter(
            Q(name__icontains=query) | Q(location__icontains=query)
        )
        
    if request.headers.get('HX-Request'):
        return render(request, 'admin/partials/company_table_body.html', {'companies': companies})
        
    return render(request, 'admin/company_list.html', {'companies': companies, 'query': query})


@user_passes_test(is_staff)
def admin_company_edit(request, pk):
    company = get_object_or_404(Company, pk=pk)
    form = CompanyForm(request.POST or None, request.FILES or None, instance=company)
    if request.method == 'POST' and form.is_valid():
        form.save()
        return redirect('admin_company_list')
    return render(request, 'admin/company_form.html', {'form': form, 'company': company})


@user_passes_test(is_staff)
def admin_company_delete(request, pk):
    company = get_object_or_404(Company, pk=pk)
    if request.method == 'POST':
        company.delete()
        return redirect('admin_company_list')
    return render(request, 'admin/company_confirm_delete.html', {'company': company})


@user_passes_test(is_staff)
@csrf_exempt
def admin_company_toggle_active(request, pk):
    company = get_object_or_404(Company, pk=pk)
    company.is_active = not company.is_active
    company.save()
    
    status_class = "bg-primary/10 text-primary" if company.is_active else "bg-surface-container-high text-outline"
    status_text = "On" if company.is_active else "Off"
    
    return render(request, 'admin/partials/company_status_badge.html', {
        'company': company,
        'status_class': status_class,
        'status_text': status_text
    })


@user_passes_test(is_staff)
@csrf_exempt
def admin_company_update_deadline(request, pk):
    company = get_object_or_404(Company, pk=pk)
    deadline_str = request.POST.get('subscription_deadline')
    
    if deadline_str:
        try:
            from django.utils.dateparse import parse_datetime
            from django.utils import timezone
            if len(deadline_str) == 10: # YYYY-MM-DD
                deadline_str += " 00:00:00"
            
            deadline = timezone.make_aware(parse_datetime(deadline_str))
            company.subscription_deadline = deadline
            company.save()
            return JsonResponse({'status': 'ok'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    else:
        company.subscription_deadline = None
        company.save()
        return JsonResponse({'status': 'ok'})


@user_passes_test(is_staff)
def admin_banner_list(request):
    banners = HomeBanner.objects.all().order_by('order', '-created_at')
    return render(request, 'admin/banner_list.html', {'banners': banners})


@user_passes_test(is_staff)
def admin_banner_create(request):
    form = HomeBannerForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        form.save()
        return redirect('admin_banner_list')
    return render(request, 'admin/banner_form.html', {'form': form})


@user_passes_test(is_staff)
def admin_banner_edit(request, pk):
    banner = get_object_or_404(HomeBanner, pk=pk)
    form = HomeBannerForm(request.POST or None, request.FILES or None, instance=banner)
    if request.method == 'POST' and form.is_valid():
        form.save()
        return redirect('admin_banner_list')
    return render(request, 'admin/banner_form.html', {'form': form, 'banner': banner})


@user_passes_test(is_staff)
def admin_banner_delete(request, pk):
    banner = get_object_or_404(HomeBanner, pk=pk)
    if request.method == 'POST':
        banner.delete()
        return redirect('admin_banner_list')
    return render(request, 'admin/banner_confirm_delete.html', {'banner': banner})
