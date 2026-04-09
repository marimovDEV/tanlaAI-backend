from django.core.paginator import Paginator
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from django.db.models import Q
from django.http import HttpResponse
from django.urls import reverse
from shop.models import Product, Category, TelegramUser, Company, HomeBanner, Wishlist
from shop.forms import ProductForm


def home_view(request):
    category_id = request.GET.get('category')
    categories = Category.objects.all()
    now = timezone.now()
    active_company_q = Q(company__is_active=True) & (Q(company__subscription_deadline__gt=now) | Q(company__subscription_deadline__isnull=True))
    
    if category_id:
        products = Product.objects.filter(active_company_q, category_id=category_id).select_related('category', 'company', 'owner')
    else:
        products = Product.objects.filter(active_company_q).select_related('category', 'company', 'owner')
    
    tg_user = None
    tg_user_id = request.session.get('tg_user_id')
    if tg_user_id:
        tg_user = TelegramUser.objects.filter(id=tg_user_id).first()
        
    banners = HomeBanner.objects.filter(is_active=True).order_by('order', '-created_at')
    
    context = {
        'products': products,
        'categories': categories,
        'active_category': category_id,
        'tg_user': tg_user,
        'banners': banners,
    }
    return render(request, 'home.html', context)


def product_detail_view(request, pk):
    product = get_object_or_404(Product, pk=pk)
    is_wishlisted = False
    tg_user_id = request.session.get('tg_user_id')
    if tg_user_id:
        is_wishlisted = Wishlist.objects.filter(user_id=tg_user_id, product=product).exists()
    return render(request, 'product_detail.html', {
        'product': product,
        'is_wishlisted': is_wishlisted,
    })


def search_view(request):
    query = request.GET.get('q')
    now = timezone.now()
    active_company_q = Q(company__is_active=True) & (Q(company__subscription_deadline__gt=now) | Q(company__subscription_deadline__isnull=True))

    if query:
        products = Product.objects.filter(active_company_q).filter(
            Q(name__icontains=query) | Q(description__icontains=query)
        ).select_related('category', 'company', 'owner')
    else:
        products = Product.objects.filter(active_company_q).order_by('?').select_related('category', 'company', 'owner')
    
    paginator = Paginator(products, 30)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    elided_page_range = paginator.get_elided_page_range(number=page_obj.number, on_each_side=1, on_ends=1)
    
    context = {
        'products': page_obj,
        'page_obj': page_obj,
        'query': query,
        'elided_page_range': elided_page_range,
    }
    
    if request.headers.get('HX-Request') and not request.headers.get('HX-Boosted'):
        return render(request, 'partials/product_grid.html', context)
        
    return render(request, 'search.html', context)


def discounts_view(request):
    query = request.GET.get('q')
    company_id = request.GET.get('company')
    now = timezone.now()
    active_company_q = Q(company__is_active=True) & (Q(company__subscription_deadline__gt=now) | Q(company__subscription_deadline__isnull=True))
    
    products = Product.objects.filter(active_company_q, is_on_sale=True).select_related('category', 'company', 'owner')
    products = products.filter(Q(sale_end_date__gt=now) | Q(sale_end_date__isnull=True))

    if query:
        products = products.filter(Q(name__icontains=query) | Q(description__icontains=query))
    
    if company_id:
        products = products.filter(company_id=company_id)
        
    products = products.order_by('-id')
    participating_companies = Company.objects.filter(products__is_on_sale=True).distinct()

    paginator = Paginator(products, 20)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    elided_page_range = paginator.get_elided_page_range(number=page_obj.number, on_each_side=1, on_ends=1)

    context = {
        'products': page_obj,
        'page_obj': page_obj,
        'query': query,
        'selected_company': company_id,
        'participating_companies': participating_companies,
        'elided_page_range': elided_page_range,
    }

    if request.headers.get('HX-Request') and not request.headers.get('HX-Boosted'):
        return render(request, 'partials/product_grid.html', context)
        
    return render(request, 'discounts.html', context)


def product_create_view(request):
    tg_user_id = request.session.get('tg_user_id')
    if not tg_user_id:
        return redirect('home')
    
    tg_user = get_object_or_404(TelegramUser, pk=tg_user_id)
    company = getattr(tg_user, 'company', None)
    
    if not company:
        return render(request, 'product_create_error.html')
    
    # Check subscription limits
    from shop.models import Subscription
    subscription, _ = Subscription.objects.get_or_create(company=company)
    current_count = company.products.count()
    
    if current_count >= subscription.max_products:
        return render(request, 'product_create_error.html', {
            'error': 'limit_reached',
            'limit': subscription.max_products
        })
    
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES)
        if form.is_valid():
            product = form.save(commit=False)
            product.owner = tg_user
            product.company = company
            product.save()
            return redirect('home')
    else:
        form = ProductForm()
    
    return render(request, 'product_create.html', {'form': form, 'company': company})


def product_edit_view(request, pk):
    tg_user_id = request.session.get('tg_user_id')
    if not tg_user_id:
        return redirect('home')
    
    tg_user = get_object_or_404(TelegramUser, pk=tg_user_id)
    product = get_object_or_404(Product, pk=pk)
    
    is_admin = request.user.is_staff
    is_owner = (product.owner == tg_user)
    
    if not (is_admin or is_owner):
        return redirect('product_detail', pk=pk)

    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES, instance=product)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = ProductForm(instance=product)
    
    return render(request, 'product_create.html', {
        'form': form, 
        'company': product.company, 
        'is_edit': True,
        'product': product
    })


def product_delete_view(request, pk):
    tg_user_id = request.session.get('tg_user_id')
    if not tg_user_id:
        return redirect('home')
    
    tg_user = get_object_or_404(TelegramUser, pk=tg_user_id)
    product = get_object_or_404(Product, pk=pk)
    
    is_admin = request.user.is_staff
    is_owner = (product.owner == tg_user)
    
    if not (is_admin or is_owner):
        return redirect('product_detail', pk=pk)

    if request.method == 'POST':
        product.delete()
        
        if request.headers.get('HX-Request'):
            response = HttpResponse()
            response['HX-Redirect'] = reverse('home')
            return response
            
        return redirect('home')
    
    return redirect('product_edit', pk=pk)
