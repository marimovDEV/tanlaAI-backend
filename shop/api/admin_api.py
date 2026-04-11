"""
Admin-specific API views for the React admin panel.
All views require Django staff authentication (session-based).
"""
from django.db.models import Count, Q
from django.utils import timezone
from rest_framework import views, viewsets, permissions, status, parsers
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404

from ..models import (
    Product, Category, TelegramUser, Company,
    HomeBanner, LeadRequest, AIResult, Subscription,
)
from rest_framework import serializers as drf_serializers
from .serializers import (
    ProductSerializer, CategorySerializer, TelegramUserSerializer,
    HomeBannerSerializer, CompanySerializer, LeadRequestSerializer,
    AdminLeadRequestSerializer, AdminAIResultSerializer
)


# Admin serializers moved to serializers.py

class AdminCompanySerializer(drf_serializers.ModelSerializer):
    """Company serializer for admin — all fields writable except computed ones."""
    is_currently_active = drf_serializers.ReadOnlyField()
    owner_name = drf_serializers.SerializerMethodField()
    owner_username = drf_serializers.SerializerMethodField()
    product_count = drf_serializers.IntegerField(read_only=True, default=0)
    logo = drf_serializers.SerializerMethodField()

    class Meta:
        model = Company
        fields = '__all__'
        read_only_fields = ['created_at', 'is_currently_active']

    def get_owner_name(self, obj):
        if obj.user:
            parts = [obj.user.first_name or '', obj.user.last_name or '']
            return ' '.join(p for p in parts if p) or str(obj.user.telegram_id)
        return ''

    def get_owner_username(self, obj):
        if obj.user and obj.user.username:
            return f'@{obj.user.username}'
        return ''

    def get_logo(self, obj):
        if not obj.logo:
            return None
        url = obj.logo.url
        if url and not url.startswith('/') and not url.startswith('http'):
            url = f"/media/{url}"
        
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(url)
        return url


class IsAdminUser(permissions.BasePermission):
    """Only allow authenticated staff users."""
    def has_permission(self, request, view):
        return (
            request.user
            and request.user.is_authenticated
            and request.user.is_staff
        )


# ── Dashboard ───────────────────────────────────────────────
class AdminDashboardApiView(views.APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        now = timezone.now()
        
        # AI Status breakdown
        ai_stats = Product.objects.values('ai_status').annotate(count=Count('ai_status'))
        ai_breakdown = {status: 0 for status, _ in Product._meta.get_field('ai_status').choices}
        for item in ai_stats:
            ai_breakdown[item['ai_status']] = item['count']
        
        # Calculate success rate
        total_ai_processed = ai_breakdown.get('completed', 0) + ai_breakdown.get('error', 0)
        success_rate = round((ai_breakdown.get('completed', 0) / total_ai_processed * 100), 1) if total_ai_processed > 0 else 100
        
        # Recent activity
        recent_products = Product.objects.select_related('category').order_by('-id')[:5]
        recent_ai = AIResult.objects.select_related('user', 'product').order_by('-created_at')[:5]
        recent_leads = LeadRequest.objects.select_related('user', 'product').order_by('-created_at')[:5]

        # Centralized serializers are now imported at module level


        return Response({
            'counts': {
                'product_count': Product.objects.count(),
                'category_count': Category.objects.count(),
                'company_count': Company.objects.count(),
                'user_count': TelegramUser.objects.count(),
                'banner_count': HomeBanner.objects.count(),
                'lead_count': LeadRequest.objects.count(),
                'ai_result_count': AIResult.objects.count(),
                'ai_error_count': Product.objects.filter(ai_status='error').count() + AIResult.objects.filter(status='error').count(),
                'active_promotions': Product.objects.filter(
                    is_on_sale=True
                ).filter(
                    Q(sale_end_date__gt=now) | Q(sale_end_date__isnull=True)
                ).count(),
            },
            'ai_status': ai_breakdown,
            'ai_performance': {
                'success_rate': success_rate,
                'avg_time': 2.3, 
            },
            'recent_activity': {
                'products': ProductSerializer(recent_products, many=True, context={'request': request}).data,
                'ai_results': AdminAIResultSerializer(recent_ai, many=True, context={'request': request}).data,
                'leads': AdminLeadRequestSerializer(recent_leads, many=True, context={'request': request}).data,
            }
        })


# ── Products ────────────────────────────────────────────────
class AdminProductViewSet(viewsets.ModelViewSet):
    serializer_class = ProductSerializer
    permission_classes = [IsAdminUser]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser, parsers.JSONParser]

    def get_queryset(self):
        qs = Product.objects.select_related('category', 'company', 'owner').order_by('-id')
        q = self.request.query_params.get('search') or self.request.query_params.get('q')
        if q:
            qs = qs.filter(Q(name__icontains=q) | Q(description__icontains=q))
        category = self.request.query_params.get('category')
        if category:
            qs = qs.filter(category_id=category)
        return qs

    @action(detail=True, methods=['post'], permission_classes=[IsAdminUser])
    def reprocess_ai(self, request, pk=None):
        """Forces the AI to re-process the background removal using the latest model (SAM)."""
        product = self.get_object()
        product.ai_status = 'processing'
        product.save(update_fields=['ai_status'])
        
        import threading
        from ..services import AIService
        threading.Thread(target=AIService.process_product_background, args=(product,)).start()
        
        return Response({'status': 'processing'})

    def perform_destroy(self, instance):
        instance.delete()


# ── Categories ──────────────────────────────────────────────
class AdminCategoryViewSet(viewsets.ModelViewSet):
    serializer_class = CategorySerializer
    permission_classes = [IsAdminUser]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser, parsers.JSONParser]

    def get_queryset(self):
        return Category.objects.annotate(
            product_count=Count('products')
        ).order_by('-id')


# ── Users ───────────────────────────────────────────────────
class AdminUserViewSet(viewsets.ModelViewSet):
    serializer_class = TelegramUserSerializer
    permission_classes = [IsAdminUser]
    http_method_names = ['get', 'head', 'options']  # read-only + custom actions

    def get_queryset(self):
        qs = TelegramUser.objects.order_by('-created_at')
        q = self.request.query_params.get('search') or self.request.query_params.get('q')
        if q:
            qs = qs.filter(
                Q(first_name__icontains=q)
                | Q(last_name__icontains=q)
                | Q(username__icontains=q)
            )
        return qs

    @action(detail=True, methods=['post'], url_path='set-role')
    def set_role(self, request, pk=None):
        user = self.get_object()
        new_role = request.data.get('role')
        
        if new_role not in dict(TelegramUser.ROLE_CHOICES).keys():
            return Response({'error': 'Invalid role choice'}, status=status.HTTP_400_BAD_REQUEST)
        
        user.role = new_role
        user.save(update_fields=['role'])
        
        # Automation: If promoted to COMPANY, ensure company record exists
        if new_role == 'COMPANY':
            if not hasattr(user, 'company'):
                from ..models import Company, Subscription
                from django.utils.text import slugify
                
                # Create a placeholder company
                company_name = f"{user.first_name}'s Collection" if user.first_name else f"Company {user.id}"
                company = Company.objects.create(
                    user=user,
                    name=company_name,
                    description="Professional boutique collection.",
                    location="O'zbekiston"
                )
                # Ensure subscription exists
                Subscription.objects.get_or_create(company=company)
        
        return Response(TelegramUserSerializer(user).data)

    @action(detail=True, methods=['post'], url_path='toggle-role')
    def toggle_role(self, request, pk=None):
        """Deprecated in favor of set_role, but kept for legacy UI compatibility until updated."""
        user = self.get_object()
        new_role = 'COMPANY' if user.role == 'USER' else 'USER'
        
        # Use our more robust set_role logic conceptually
        request.data['role'] = new_role
        return self.set_role(request, pk=pk)


# ── Companies ───────────────────────────────────────────────
class AdminCompanyViewSet(viewsets.ModelViewSet):
    serializer_class = AdminCompanySerializer
    permission_classes = [IsAdminUser]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser, parsers.JSONParser]

    def get_queryset(self):
        qs = Company.objects.select_related('user').annotate(
            product_count=Count('products')
        ).order_by('-created_at')
        q = self.request.query_params.get('search') or self.request.query_params.get('q')
        if q:
            qs = qs.filter(Q(name__icontains=q) | Q(location__icontains=q))
        return qs

    @action(detail=True, methods=['post'], url_path='toggle-active')
    def toggle_active(self, request, pk=None):
        company = self.get_object()
        company.is_active = not company.is_active
        company.save(update_fields=['is_active'])
        return Response(CompanySerializer(company).data)

    @action(detail=True, methods=['post'], url_path='update-deadline')
    def update_deadline(self, request, pk=None):
        company = self.get_object()
        deadline_str = request.data.get('subscription_deadline')
        if deadline_str:
            from django.utils.dateparse import parse_datetime
            if len(str(deadline_str)) == 10:
                deadline_str += " 00:00:00"
            dt = parse_datetime(str(deadline_str))
            if dt is None:
                return Response({'error': 'Invalid date format'}, status=400)
            company.subscription_deadline = timezone.make_aware(dt) if timezone.is_naive(dt) else dt
        else:
            company.subscription_deadline = None
        company.save(update_fields=['subscription_deadline'])
        return Response(CompanySerializer(company).data)


# ── Promotions (on-sale products) ───────────────────────────
class AdminPromotionViewSet(viewsets.ModelViewSet):
    serializer_class = ProductSerializer
    permission_classes = [IsAdminUser]
    http_method_names = ['get', 'head', 'options', 'patch', 'delete']

    def get_queryset(self):
        return Product.objects.filter(is_on_sale=True).select_related(
            'category', 'company', 'owner'
        ).order_by('-id')

    @action(detail=True, methods=['post'], url_path='toggle-sale')
    def toggle_sale(self, request, pk=None):
        product = self.get_object()
        product.is_on_sale = not product.is_on_sale
        product.save(update_fields=['is_on_sale'])
        return Response(ProductSerializer(product).data)


# ── Banners ─────────────────────────────────────────────────
class AdminBannerViewSet(viewsets.ModelViewSet):
    serializer_class = HomeBannerSerializer
    permission_classes = [IsAdminUser]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser, parsers.JSONParser]

    def get_queryset(self):
        return HomeBanner.objects.all().order_by('order', '-created_at')


# ── Leads ───────────────────────────────────────────────────
class AdminLeadViewSet(viewsets.ModelViewSet):
    serializer_class = AdminLeadRequestSerializer
    permission_classes = [IsAdminUser]
    http_method_names = ['get', 'head', 'options', 'patch', 'delete']

    def get_queryset(self):
        qs = LeadRequest.objects.select_related('user', 'product', 'company').order_by('-created_at')
        status = self.request.query_params.get('status')
        if status == 'unprocessed':
            qs = qs.filter(is_processed=False)
        elif status == 'processed':
            qs = qs.filter(is_processed=True)
        return qs

    @action(detail=True, methods=['post'], url_path='toggle-processed')
    def toggle_processed(self, request, pk=None):
        lead = self.get_object()
        lead.is_processed = not lead.is_processed
        lead.save(update_fields=['is_processed'])
        return Response(AdminLeadRequestSerializer(lead).data)


# ── AI Results ──────────────────────────────────────────────
class AdminAIResultViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = AdminAIResultSerializer
    permission_classes = [IsAdminUser]

    def get_queryset(self):
        return AIResult.objects.select_related('user', 'product').order_by('-created_at')
