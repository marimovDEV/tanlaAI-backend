import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from shop.models import TelegramUser, SystemSetting
from shop.utils import verify_telegram_webapp_data


def is_staff(user):
    return user.is_staff


@csrf_exempt
def auth_login(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            init_data = data.get('initData')

            db_token = SystemSetting.get_solo().telegram_bot_token
            active_bot_token = db_token or settings.TELEGRAM_BOT_TOKEN
            user_data = verify_telegram_webapp_data(init_data, active_bot_token)
            if user_data:
                telegram_id = user_data.get('id')
                first_name = user_data.get('first_name')
                last_name = user_data.get('last_name')
                username = user_data.get('username')
                photo_url = user_data.get('photo_url')
                
                user, created = TelegramUser.objects.update_or_create(
                    telegram_id=telegram_id,
                    defaults={
                        'first_name': first_name,
                        'last_name': last_name,
                        'username': username,
                        'photo_url': photo_url,
                    }
                )
                
                # Set user in session
                request.session['tg_user_id'] = user.id
                request.session.modified = True
                return JsonResponse({'status': 'ok', 'user_id': user.id})
            
            return JsonResponse({'status': 'error', 'message': 'Invalid data'}, status=403)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Only POST allowed'}, status=405)
