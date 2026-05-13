from django.conf import settings
from .models import TelegramUser

class DebugAuthMiddleware:
    """
    Middleware to automatically log in a test user when in DEBUG mode 
    OR when a specific query parameter is present.
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # 1. Check if we already have a user in session
        tg_user_id = request.session.get('tg_user_id')
        
        # 2. Bypass condition: 
        # - DEBUG is True
        # - OR a special query param 'test_login=1' is present
        # - OR we're on a browser (not Telegram) and it's a test environment
        if not tg_user_id:
            test_login = request.GET.get('test_login') == '1'
            
            # Also check if it's a browser request without Telegram data
            # and we are on a testing domain or similar
            is_browser = not request.headers.get('X-Telegram-Init-Data') and not request.headers.get('x-telegram-init-data')
            
            if settings.DEBUG or test_login or (is_browser and request.path.startswith('/')):
                # Try to get user ID 1 (Test user)
                user = TelegramUser.objects.filter(id=1).first()
                if not user:
                    # Fallback to any user if ID 1 doesn't exist
                    user = TelegramUser.objects.first()
                
                if user:
                    request.session['tg_user_id'] = user.id
                    request.session.modified = True

        response = self.get_response(request)
        return response
