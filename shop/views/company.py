from django.shortcuts import render, get_object_or_404, redirect
from django.db.models import Count, F, Q
from shop.models import Company, TelegramUser
from shop.forms import CompanyForm


def leaders_view(request):
    query = request.GET.get('q')
    
    # Score-based ranking: products + AI visualizations + wishlist saves
    companies = Company.objects.annotate(
        product_count=Count('products', distinct=True),
        ai_usage=Count('products__ai_visualizations', distinct=True),
        wishlist_count=Count('products__wishlisted_by', distinct=True),
    ).annotate(
        score=F('product_count') + F('ai_usage') + F('wishlist_count')
    )
    
    if query:
        companies = companies.filter(Q(name__icontains=query) | Q(location__icontains=query))
    
    companies = companies.order_by('-score', '-created_at')
    
    return render(request, 'leaders.html', {'companies': companies, 'query': query})


def company_detail_view(request, pk):
    company = get_object_or_404(Company, pk=pk)
    if not company.is_currently_active:
        return render(request, 'company_inactive.html', {'company': company})
    
    products = company.products.all()
    return render(request, 'company_detail.html', {'company': company, 'products': products})


def company_upsert_view(request):
    tg_user_id = request.session.get('tg_user_id')
    if not tg_user_id:
        return redirect('home')
    
    tg_user = get_object_or_404(TelegramUser, pk=tg_user_id)
    company = getattr(tg_user, 'company', None)
    
    if request.method == 'POST':
        form = CompanyForm(request.POST, request.FILES, instance=company)
        if form.is_valid():
            new_company = form.save(commit=False)
            new_company.user = tg_user
            new_company.save()
            return redirect('leaders')
    else:
        form = CompanyForm(instance=company)
    
    return render(request, 'company_form.html', {'form': form, 'company': company})
