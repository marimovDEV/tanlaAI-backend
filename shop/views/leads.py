from django.shortcuts import get_object_or_404, render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from shop.models import Product, TelegramUser, LeadRequest
from shop.forms import LeadRequestForm

@csrf_exempt
def create_lead_request(request, pk):
    """View to handle lead generation (call/telegram/measurement requests)."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)
    
    product = get_object_or_404(Product, pk=pk)
    tg_user_id = request.session.get('tg_user_id')
    
    if not tg_user_id:
        return JsonResponse({'status': 'error', 'message': 'Authentication required'}, status=401)
    
    tg_user = get_object_or_404(TelegramUser, pk=tg_user_id)
    
    form = LeadRequestForm(request.POST)
    if form.is_valid():
        lead = form.save(commit=False)
        lead.user = tg_user
        lead.product = product
        lead.company = product.company
        lead.save()
        
        # Return success partial for HTMX
        if request.headers.get('HX-Request'):
            return render(request, 'partials/lead_success.html', {
                'lead_type': lead.lead_type,
                'product': product
            })
            
        return JsonResponse({'status': 'ok', 'lead_id': lead.id})
    
    return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
