from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from shop.models import TelegramUser, Product, AIResult, Wishlist


def profile_view(request):
    tg_user = None
    tg_user_id = request.session.get('tg_user_id')
    if tg_user_id:
        tg_user = TelegramUser.objects.filter(id=tg_user_id).first()
    
    return render(request, 'profile.html', {'tg_user': tg_user})


def wishlist_view(request):
    tg_user = None
    tg_user_id = request.session.get('tg_user_id')
    wishlist_products = []
    ai_results = []
    if tg_user_id:
        tg_user = TelegramUser.objects.filter(id=tg_user_id).first()
        if tg_user:
            ai_results = AIResult.objects.filter(user=tg_user).select_related('product')
            wishlist_products = Wishlist.objects.filter(user=tg_user).select_related('product', 'product__category', 'product__company')
    
    return render(request, 'wishlist.html', {
        'tg_user': tg_user,
        'ai_results': ai_results,
        'wishlist_products': wishlist_products,
    })


@csrf_exempt
def toggle_wishlist(request, pk):
    """HTMX endpoint to toggle wishlist status for a product."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    
    tg_user_id = request.session.get('tg_user_id')
    if not tg_user_id:
        return JsonResponse({'error': 'Not authenticated'}, status=401)
    
    tg_user = TelegramUser.objects.filter(id=tg_user_id).first()
    if not tg_user:
        return JsonResponse({'error': 'User not found'}, status=404)
    
    product = get_object_or_404(Product, pk=pk)
    
    wishlist_item, created = Wishlist.objects.get_or_create(user=tg_user, product=product)
    
    if not created:
        wishlist_item.delete()
        is_wishlisted = False
    else:
        is_wishlisted = True
    
    if request.headers.get('HX-Request'):
        return render(request, 'partials/wishlist_btn.html', {
            'product': product,
            'is_wishlisted': is_wishlisted,
        })
    
    return JsonResponse({'wishlisted': is_wishlisted})
