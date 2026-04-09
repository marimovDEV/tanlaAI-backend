from .models import TelegramUser

def tg_user_processor(request):
    # Use cached user on the request object if it was already fetched during this request lifecycle
    if hasattr(request, '_cached_tg_user'):
        return {'tg_user': request._cached_tg_user}
        
    tg_user = None
    tg_user_id = request.session.get('tg_user_id')
    if tg_user_id:
        tg_user = TelegramUser.objects.filter(id=tg_user_id).first()
        request._cached_tg_user = tg_user # Cache it for the rest of the request
    
    return {
        'tg_user': tg_user
    }

def base_template_processor(request):
    is_htmx = bool(request.headers.get('HX-Request'))
    # is_boosted = bool(request.headers.get('HX-Boosted'))
    
    # Send partial if it's an HTMX request (including boosted navigation)
    # HTMX handles the title swap automatically from the partial.
    if is_htmx:
        return { 
            'base_template': 'partial_base.html',
            'is_htmx': True
        }
    return { 
        'base_template': 'base.html',
        'is_htmx': False
    }
