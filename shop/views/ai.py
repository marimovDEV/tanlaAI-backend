import os
import uuid
from django.shortcuts import render, get_object_or_404
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from concurrent.futures import ThreadPoolExecutor
from shop.models import Product, AIResult, TelegramUser
from shop.services import AIService

# Global executor for AI tasks
ai_executor = ThreadPoolExecutor(max_workers=2)

def run_ai_background(s_key, s_data_key, p_pk, r_path, res_path, u_id):
    """
    Background worker that runs the door visualization via AIService.
    We import models inside to avoid app-loading issues in threads.
    """
    from django.contrib.sessions.backends.db import SessionStore
    from .models import Product, AIResult, TelegramUser
    from .services import AIService
    import os

    session = SessionStore(session_key=s_key)
    try:
        prod = Product.objects.get(pk=p_pk)
        
        # Run AI via service layer (includes retry logic)
        AIService.generate_room_preview(prod, r_path, res_path)
        
        # Success: update session
        data = session.get(s_data_key, {})
        data['status'] = 'done'
        session[s_data_key] = data
        
        # Save to database for profile history
        if u_id:
            try:
                user = TelegramUser.objects.get(id=u_id)
                relative_result = os.path.join('ai_results', os.path.basename(res_path))
                relative_input = os.path.join('ai_temp', os.path.basename(r_path))
                AIResult.objects.create(
                    user=user,
                    product=prod,
                    image=relative_result,
                    input_image=relative_input,
                    status='done',
                )
            except Exception as db_err:
                print(f"DEBUG: AI Record Save Error: {db_err}")

        prod.ai_status = 'completed'
        prod.save()
        print(f"DEBUG: AI Background Success for Product {p_pk}")

    except Exception as e:
        print(f"DEBUG: AI Background Error for Product {p_pk}: {e}")
        try:
            prod = Product.objects.get(pk=p_pk)
            prod.ai_status = 'error'
            prod.save()
        except:
            pass
        data = session.get(s_data_key, {})
        data['status'] = 'error'
        data['error_msg'] = str(e)
        session[s_data_key] = data
    finally:
        session.save()

@csrf_exempt
def ai_generate_view(request, pk):
    """
    AI Visualization page. Validates inputs, saves temp file, and starts AI.
    The polling endpoint /ai-generate/result/ delivers the result when ready.
    """
    product = get_object_or_404(Product, pk=pk)
    
    # Check subscription limits for AI
    from shop.models import Subscription, AIResult
    company = product.company
    subscription, _ = Subscription.objects.get_or_create(company=company)
    
    # Simple count of all successful AI generations for this company
    current_ai_count = AIResult.objects.filter(product__company=company, status='done').count()
    
    if current_ai_count >= subscription.ai_generations_limit:
        return render(request, 'ai_generate.html', {
            'product': product,
            'error': 'limit_reached',
            'limit': subscription.ai_generations_limit
        })

    if product.ai_status == 'processing':
        return render(request, 'ai_generate.html', {'product': product, 'generating': True})
    
    if product.ai_status == 'none' and product.category and product.category.name == 'Eshiklar':
        return render(request, 'ai_generate.html', {'product': product, 'error': 'not_ready'})

    if request.method == 'POST':
        room_photo = request.FILES.get('room_photo')
        user_height = request.POST.get('height', '').strip()
        user_width  = request.POST.get('width', '').strip()

        if not room_photo:
            return render(request, 'ai_generate.html', {'product': product, 'error': 'no_photo'})

        # Dimension check
        if product.height and product.width:
            TOLERANCE = 5.0
            try:
                if abs(float(user_height) - float(product.height)) > TOLERANCE or \
                   abs(float(user_width) - float(product.width)) > TOLERANCE:
                    return render(request, 'ai_generate.html', {
                        'product': product,
                        'error': 'dimension_mismatch',
                        'user_height': user_height, 'user_width': user_width,
                        'company': product.company
                    })
            except (ValueError, TypeError):
                return render(request, 'ai_generate.html', {
                    'product': product,
                    'error': 'invalid_dimensions',
                    'user_height': user_height, 'user_width': user_width,
                    'company': product.company
                })

        request_id = str(uuid.uuid4())
        room_path = os.path.join(settings.MEDIA_ROOT, 'ai_temp', f'{request_id}.png')
        result_path = os.path.join(settings.MEDIA_ROOT, 'ai_results', f'{request_id}.png')
        
        # Save uploaded room photo to disk
        with open(room_path, 'wb+') as destination:
            for chunk in room_photo.chunks():
                destination.write(chunk)

        # Ensure session exists
        if not request.session.session_key:
            request.session.create()
        tg_user_id = request.session.get('tg_user_id')
        
        session_key_data = f'ai_gen_{pk}'
        request.session[session_key_data] = {
            'status': 'running',
            'request_id': request_id, 
        }
        request.session.modified = True
        request.session.save()
        
        # Save DB status
        product.ai_status = 'processing'
        product.save()

        # Submit to background executor
        ai_executor.submit(
            run_ai_background, 
            request.session.session_key, 
            session_key_data, 
            product.pk, 
            room_path, 
            result_path, 
            tg_user_id
        )

        return render(request, 'ai_generate.html', {
            'product': product,
            'generating': True,
            'user_height': user_height,
            'user_width': user_width,
        })

    return render(request, 'ai_generate.html', {'product': product})


def ai_generate_poll(request, pk):
    """HTMX polling endpoint — returns the result via media URL when file is ready."""
    product = get_object_or_404(Product, pk=pk)
    
    # Priority 1: Check database status
    if product.ai_status == 'completed':
        # Get the latest result for this product
        latest_result = AIResult.objects.filter(product=product).order_by('-created_at').first()
        if latest_result:
            return render(request, 'partials/ai_result.html', {
                'product': product,
                'result_url': latest_result.image.url,
            })
    
    if product.ai_status == 'error':
        return render(request, 'partials/ai_result.html', {
            'product': product,
            'error': 'ai_failed',
        })

    # Priority 2: Fallback to session check for temporary paths
    session_key = f'ai_gen_{pk}'
    data = request.session.get(session_key, {})
    request_id = data.get('request_id')

    if request_id:
        result_path = os.path.join(settings.MEDIA_ROOT, 'ai_results', f'{request_id}.png')
        if os.path.exists(result_path):
            result_url = f'{settings.MEDIA_URL}ai_results/{request_id}.png'
            return render(request, 'partials/ai_result.html', {
                'product': product,
                'result_url': result_url,
            })
    
    # If still processing or no data found, keep polling
    return render(request, 'partials/ai_result.html', {'product': product, 'pending': True})
