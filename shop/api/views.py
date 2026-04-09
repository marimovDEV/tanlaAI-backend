import os
import uuid
from concurrent.futures import ThreadPoolExecutor
import shlex
import subprocess

from django.db import models
from django.utils import timezone
from rest_framework import viewsets, permissions, status, views
from rest_framework.response import Response
from rest_framework.decorators import action
from django.conf import settings
from django.shortcuts import get_object_or_404
from django.contrib.auth import authenticate, login, logout
from rest_framework.exceptions import PermissionDenied, ValidationError
from ..models import (
    AIResult, Category, Company, HomeBanner, LeadRequest,
    Product, Subscription, TelegramUser, Wishlist, SystemSetting,
)
from .serializers import (
    AIResultSerializer, CategorySerializer, CompanySerializer,
    HomeBannerSerializer, LeadRequestSerializer, ProductSerializer,
    TelegramUserSerializer, WishlistSerializer,
)
from ..services import AIService
from ..utils import verify_telegram_webapp_data


ai_executor = ThreadPoolExecutor(max_workers=2)


def run_api_ai_background(product_id, room_path, result_path, tg_user_id):
    try:
        product = Product.objects.get(pk=product_id)
        AIService.generate_room_preview(product, room_path, result_path)

        if tg_user_id:
            user = TelegramUser.objects.filter(id=tg_user_id).first()
            if user:
                AIResult.objects.create(
                    user=user,
                    product=product,
                    image=os.path.join('ai_results', os.path.basename(result_path)),
                    input_image=os.path.join('ai_temp', os.path.basename(room_path)),
                    status='done',
                )

        product.ai_status = 'completed'
        product.save(update_fields=['ai_status'])
    except Exception as error:
        Product.objects.filter(pk=product_id).update(ai_status='error')
        print(f"DEBUG: API AI generation error for product {product_id}: {error}")


def get_tg_user(request):
    tg_user_id = request.session.get('tg_user_id')
    
    # Fallback to header-based authentication and auto-registration
    if not tg_user_id:
        init_data = request.headers.get('X-Telegram-Init-Data')
        if init_data:
            user_data = verify_telegram_webapp_data(init_data, settings.TELEGRAM_BOT_TOKEN)
            if user_data:
                # Auto-sync/register user
                user_id = user_data.get('id')
                user, created = TelegramUser.objects.update_or_create(
                    id=user_id,
                    defaults={
                        'first_name': user_data.get('first_name', ''),
                        'last_name': user_data.get('last_name', ''),
                        'username': user_data.get('username'),
                        'language_code': user_data.get('language_code'),
                    }
                )
                tg_user_id = user.id
                # Update session for subsequent requests
                request.session['tg_user_id'] = tg_user_id
    
    if not tg_user_id:
        return None
    return TelegramUser.objects.filter(id=tg_user_id).first()


def require_tg_user(request):
    tg_user = get_tg_user(request)
    if tg_user is None:
        raise ValidationError({'detail': 'Authentication required'})
    return tg_user


def ensure_product_owner(request, product):
    tg_user = require_tg_user(request)
    if request.user.is_staff or product.owner_id == tg_user.id:
        return tg_user
    raise PermissionDenied('You do not have permission to manage this product.')


def ensure_company_owner(request, company):
    tg_user = require_tg_user(request)
    if request.user.is_staff or company.user_id == tg_user.id:
        return tg_user
    raise PermissionDenied('You do not have permission to manage this company.')


class TelegramAuthView(views.APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        if settings.DEBUG:
            # Mock login for local browser testing
            user_data = {
                'id': 123456789,
                'first_name': 'Test',
                'last_name': 'User',
                'username': 'testuser',
                'photo_url': 'https://via.placeholder.com/150'
            }
            return self._process_user(request, user_data)
        return Response({'error': 'Method not allowed'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

    def post(self, request):
        init_data = request.data.get('initData')
        if not init_data:
            return Response({'error': 'No initData provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        user_data = verify_telegram_webapp_data(init_data, settings.TELEGRAM_BOT_TOKEN)
        if not user_data:
            return Response({'error': 'Invalid Telegram data'}, status=status.HTTP_403_FORBIDDEN)
        
        return self._process_user(request, user_data)
        
    def _process_user(self, request, user_data):
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
        
        # We'll use DRF sessions for now as it's easier to integrate with existing logic
        request.session['tg_user_id'] = user.id
        request.session.modified = True
        
        serializer = TelegramUserSerializer(user)
        return Response({
            'user': serializer.data,
            'status': 'ok'
        })

class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Category.objects.order_by('id')
    serializer_class = CategorySerializer
    permission_classes = [permissions.AllowAny]

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    permission_classes = [permissions.AllowAny]
    
    def get_queryset(self):
        queryset = Product.objects.select_related('category', 'company', 'owner').all()
        category = self.request.query_params.get('category')
        company = self.request.query_params.get('company')
        is_featured = self.request.query_params.get('is_featured')
        is_on_sale = self.request.query_params.get('is_on_sale')
        search = self.request.query_params.get('search')
        now = timezone.now()

        if self.action == 'list':
            # Show products that either have no company (system products) 
            # OR belong to an active company with a valid subscription
            queryset = queryset.filter(
                models.Q(company__isnull=True) |
                (
                    models.Q(company__is_active=True) &
                    (models.Q(company__subscription_deadline__gt=now) | models.Q(company__subscription_deadline__isnull=True))
                )
            )

        if category:
            queryset = queryset.filter(category_id=category)
        if company:
            queryset = queryset.filter(company_id=company)
        if is_featured:
            queryset = queryset.filter(is_featured=True)
        if is_on_sale:
            queryset = queryset.filter(is_on_sale=True).filter(
                models.Q(sale_end_date__gt=now) |
                models.Q(sale_end_date__isnull=True)
            )
        if search:
            queryset = queryset.filter(
                models.Q(name__icontains=search) |
                models.Q(description__icontains=search)
            )

        return queryset.order_by('-id')

    @action(detail=False, methods=['get'])
    def my(self, request):
        tg_user = get_tg_user(request)
        if tg_user is None:
            return Response({'error': 'Not authenticated'}, status=status.HTTP_401_UNAUTHORIZED)
        
        products = Product.objects.filter(owner_id=tg_user.id).select_related('category', 'company', 'owner')
        serializer = self.get_serializer(products, many=True)
        return Response(serializer.data)

    def perform_create(self, serializer):
        tg_user = require_tg_user(self.request)
        company = getattr(tg_user, 'company', None)
        if company is None:
            raise ValidationError({'detail': 'Create a company profile before adding products.'})

        subscription, _ = Subscription.objects.get_or_create(company=company)
        if company.products.count() >= subscription.max_products:
            raise ValidationError({'detail': f'You have reached your plan limit of {subscription.max_products} products.'})

        serializer.save(owner=tg_user, company=company)

    def perform_update(self, serializer):
        ensure_product_owner(self.request, serializer.instance)
        serializer.save()

    def perform_destroy(self, instance):
        ensure_product_owner(self.request, instance)
        instance.delete()

    @action(detail=True, methods=['post'], permission_classes=[permissions.AllowAny])
    def toggle_wishlist(self, request, pk=None):
        product = self.get_object()
        tg_user = get_tg_user(request)
        if tg_user is None:
            return Response({'error': 'Not authenticated'}, status=status.HTTP_401_UNAUTHORIZED)

        wishlist_item = Wishlist.objects.filter(user=tg_user, product=product).first()
        
        if wishlist_item:
            wishlist_item.delete()
            return Response({'status': 'removed'})
        else:
            Wishlist.objects.create(user=tg_user, product=product)
            return Response({'status': 'added'})

    @action(detail=True, methods=['post'], url_path='ai-generate', permission_classes=[permissions.AllowAny])
    def ai_generate(self, request, pk=None):
        product = self.get_object()
        tg_user = get_tg_user(request)

        if tg_user is None:
            return Response({'status': 'error', 'message': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)

        # Allow admin-added products without a company
        if product.company:
            subscription, _ = Subscription.objects.get_or_create(company=product.company)
            current_ai_count = AIResult.objects.filter(product__company=product.company, status='done').count()
            if current_ai_count >= subscription.ai_generations_limit:
                return Response({
                    'status': 'error',
                    'message': f'AI generation limit reached ({subscription.ai_generations_limit}).',
                    'code': 'limit_reached',
                    'limit': subscription.ai_generations_limit,
                }, status=status.HTTP_400_BAD_REQUEST)

        if product.ai_status == 'processing':
            return Response({'status': 'processing'})

        # If background is not removed yet, trigger it now instead of failing
        if product.ai_status == 'none':
            from ..signals import trigger_ai_processing
            # We trigger the processing manually (in a background thread via the signal or service)
            # For immediate UX, we tell the user to wait a bit
            product.ai_status = 'processing' # Set to processing to avoid double trigger
            product.save(update_fields=['ai_status'])
            
            # Use the service to start background removal
            from ..services import AIService
            import threading
            threading.Thread(target=AIService.process_product_background, args=(product,)).start()
            
            return Response({
                'status': 'preparing',
                'message': 'Orqa fon o‘chirilmoqda. Iltimos 10-15 soniya kutib qayta urinib ko‘ring.',
                'code': 'not_ready',
            }, status=status.HTTP_200_OK) # Return 200 instead of 400 for better UX

        room_photo = request.FILES.get('room_photo') or request.FILES.get('input_image')
        user_height = str(request.data.get('height', '')).strip()
        user_width = str(request.data.get('width', '')).strip()

        if not room_photo:
            return Response({'status': 'error', 'message': 'Please upload a room photo.', 'code': 'no_photo'}, status=status.HTTP_400_BAD_REQUEST)

        if product.height and product.width:
            tolerance = 5.0
            try:
                if abs(float(user_height) - float(product.height)) > tolerance or abs(float(user_width) - float(product.width)) > tolerance:
                    return Response({
                        'status': 'error',
                        'message': 'Door dimensions do not match the selected product.',
                        'code': 'dimension_mismatch',
                    }, status=status.HTTP_400_BAD_REQUEST)
            except (TypeError, ValueError):
                return Response({
                    'status': 'error',
                    'message': 'Please enter valid numeric dimensions.',
                    'code': 'invalid_dimensions',
                }, status=status.HTTP_400_BAD_REQUEST)

        request_id = str(uuid.uuid4())
        room_dir = os.path.join(settings.MEDIA_ROOT, 'ai_temp')
        result_dir = os.path.join(settings.MEDIA_ROOT, 'ai_results')
        os.makedirs(room_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        room_path = os.path.join(room_dir, f'{request_id}.png')
        result_path = os.path.join(result_dir, f'{request_id}.png')

        with open(room_path, 'wb+') as destination:
            for chunk in room_photo.chunks():
                destination.write(chunk)

        product.ai_status = 'processing'
        product.save(update_fields=['ai_status'])

        ai_executor.submit(run_api_ai_background, product.id, room_path, result_path, tg_user.id)

        return Response({'status': 'ok'})

    @action(detail=True, methods=['get'], url_path='ai-generate/result', permission_classes=[permissions.AllowAny])
    def ai_generate_result(self, request, pk=None):
        product = self.get_object()

        if product.ai_status == 'completed':
            latest_result = AIResult.objects.filter(product=product).order_by('-created_at').first()
            if latest_result:
                return Response({'status': 'done', 'image_url': latest_result.image.url})

        if product.ai_status == 'error':
            return Response({'status': 'error', 'message': 'AI generation failed.'})

        return Response({'status': 'processing'})

class CompanyViewSet(viewsets.ModelViewSet):
    queryset = Company.objects.all()
    serializer_class = CompanySerializer
    permission_classes = [permissions.AllowAny]
    
    def get_queryset(self):
        if self.action == 'list':
            query = self.request.query_params.get('search') or self.request.query_params.get('q')
            companies = Company.objects.filter(is_active=True).annotate(
                product_count=models.Count('products', distinct=True),
                ai_usage=models.Count('products__ai_visualizations', distinct=True),
                wishlist_count=models.Count('products__wishlisted_by', distinct=True),
            ).annotate(
                score=models.F('product_count') + models.F('ai_usage') + models.F('wishlist_count')
            )
            if query:
                companies = companies.filter(
                    models.Q(name__icontains=query) |
                    models.Q(location__icontains=query)
                )
            return companies.order_by('-score', '-created_at')
        return Company.objects.select_related('user').all()

    def perform_create(self, serializer):
        tg_user = require_tg_user(self.request)
        if hasattr(tg_user, 'company'):
            raise ValidationError({'detail': 'You already have a company profile.'})
        serializer.save(user=tg_user)

    def perform_update(self, serializer):
        ensure_company_owner(self.request, serializer.instance)
        serializer.save()

    def perform_destroy(self, instance):
        ensure_company_owner(self.request, instance)
        instance.delete()

    @action(detail=False, methods=['get', 'post', 'put', 'patch'])
    def my(self, request):
        tg_user = get_tg_user(request)
        if tg_user is None:
            return Response({'error': 'Not authenticated'}, status=status.HTTP_401_UNAUTHORIZED)

        company = Company.objects.filter(user_id=tg_user.id).first()

        if request.method == 'GET':
            if not company:
                return Response({'error': 'Company not found'}, status=status.HTTP_404_NOT_FOUND)
            serializer = self.get_serializer(company)
            return Response(serializer.data)

        partial = request.method == 'PATCH'
        serializer = self.get_serializer(company, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        serializer.save(user=tg_user)
        return Response(serializer.data, status=status.HTTP_201_CREATED if company is None else status.HTTP_200_OK)

class BannerViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = HomeBanner.objects.filter(is_active=True)
    serializer_class = HomeBannerSerializer

class WishlistViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = WishlistSerializer
    permission_classes = [permissions.AllowAny]
    
    def get_queryset(self):
        tg_user = get_tg_user(self.request)
        if tg_user is None:
            return Wishlist.objects.none()
        return Wishlist.objects.filter(user_id=tg_user.id).select_related('product', 'product__category', 'product__company')

class LeadRequestViewSet(viewsets.ModelViewSet):
    serializer_class = LeadRequestSerializer
    permission_classes = [permissions.AllowAny]
    
    def get_queryset(self):
        tg_user = get_tg_user(self.request)
        if tg_user is None:
            return LeadRequest.objects.none()
        
        # Creator can see leads related to their products or their company
        return LeadRequest.objects.filter(
            models.Q(product__owner_id=tg_user.id) | 
            models.Q(company__user_id=tg_user.id)
        ).distinct().select_related('product', 'user', 'company')

    def perform_create(self, serializer):
        tg_user = require_tg_user(self.request)
        product = serializer.validated_data['product']
        serializer.save(user=tg_user, company=product.company)

class AIResultViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = AIResultSerializer
    permission_classes = [permissions.AllowAny]
    
    def get_queryset(self):
        tg_user = get_tg_user(self.request)
        if tg_user is None:
            return AIResult.objects.none()
        return AIResult.objects.filter(user_id=tg_user.id).select_related('product', 'product__category', 'product__company')


class AdminLoginApiView(views.APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []  # Disable CSRF check for login

    def post(self, request):
        username = (request.data.get('username') or '').strip()
        password = request.data.get('password') or ''
        user = authenticate(request, username=username, password=password)
        if not user or not user.is_staff:
            return Response({'status': 'error', 'message': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)

        from rest_framework.authtoken.models import Token
        login(request, user)
        token, _ = Token.objects.get_or_create(user=user)
        return Response({
            'status': 'ok', 
            'username': user.username,
            'token': token.key
        })


class AdminLogoutApiView(views.APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []  # Disable CSRF check for logout

    def post(self, request):
        logout(request)
        return Response({'status': 'ok'})


class AdminMeApiView(views.APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        if not request.user.is_authenticated or not request.user.is_staff:
            return Response({'is_authenticated': False}, status=status.HTTP_401_UNAUTHORIZED)
        return Response({
            'is_authenticated': True,
            'username': request.user.username,
        })


class AdminSystemSettingsApiView(views.APIView):
    permission_classes = [permissions.AllowAny]

    def _check_admin(self, request):
        if not request.user.is_authenticated or not request.user.is_staff:
            raise PermissionDenied('Admin authentication required')

    def get(self, request):
        self._check_admin(request)
        setting = SystemSetting.get_solo()
        deploy_enabled = str(getattr(settings, 'ALLOW_ADMIN_DEPLOY_ACTIONS', False)).lower() in ('1', 'true', 'yes', 'on')
        return Response({
            'telegram_bot_token': setting.telegram_bot_token,
            'deploy_enabled': deploy_enabled,
        })

    def post(self, request):
        self._check_admin(request)
        setting = SystemSetting.get_solo()
        setting.telegram_bot_token = (request.data.get('telegram_bot_token') or '').strip()
        setting.save(update_fields=['telegram_bot_token', 'updated_at'])
        return Response({'status': 'ok'})


class AdminRunActionApiView(views.APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        if not request.user.is_authenticated or not request.user.is_staff:
            return Response({'status': 'error', 'message': 'Admin authentication required'}, status=status.HTTP_401_UNAUTHORIZED)

        deploy_enabled = str(getattr(settings, 'ALLOW_ADMIN_DEPLOY_ACTIONS', False)).lower() in ('1', 'true', 'yes', 'on')
        if not deploy_enabled:
            return Response({'status': 'error', 'message': 'Deploy actions disabled'}, status=status.HTTP_403_FORBIDDEN)

        action = request.data.get('action')
        command_map = {
            'git_pull': ['git', 'pull', 'origin', 'main'],
            'migrate': ['python', 'manage.py', 'migrate'],
            'collectstatic': ['python', 'manage.py', 'collectstatic', '--noinput'],
            'restart_service': shlex.split(getattr(settings, 'ADMIN_RESTART_COMMAND', 'sudo systemctl restart tanla-ai.service')),
            'status_service': shlex.split(getattr(settings, 'ADMIN_STATUS_COMMAND', 'sudo systemctl status tanla-ai.service --no-pager')),
        }
        command = command_map.get(action)
        if not command:
            return Response({'status': 'error', 'message': 'Unknown action'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            completed = subprocess.run(
                command,
                cwd=str(settings.BASE_DIR),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Response({'status': 'error', 'message': 'Command timed out after 120s'}, status=status.HTTP_504_GATEWAY_TIMEOUT)
        except Exception as exc:
            return Response({'status': 'error', 'message': str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        output = (completed.stdout or '') + (('\n' + completed.stderr) if completed.stderr else '')
        return Response({
            'status': 'ok' if completed.returncode == 0 else 'error',
            'exit_code': completed.returncode,
            'command': ' '.join(command),
            'output': output.strip(),
        })
