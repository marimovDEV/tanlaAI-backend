"""
Admin-specific API views for the React admin panel.
All views require Django staff authentication (session-based).
"""
from django.db.models import Count, Q, Sum
from django.utils import timezone
from rest_framework import views, viewsets, permissions, status, parsers
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.exceptions import PermissionDenied
from django.shortcuts import get_object_or_404

from ..models import (
    Product, Category, TelegramUser, Company,
    HomeBanner, LeadRequest, AIResult, Subscription, AITest, SystemSettings,
    Payment,
)
from rest_framework import serializers as drf_serializers
from .serializers import (
    ProductSerializer, CategorySerializer, TelegramUserSerializer,
    HomeBannerSerializer, CompanySerializer, LeadRequestSerializer,
    AdminLeadRequestSerializer, AdminAIResultSerializer, AITestSerializer,
    PaymentSerializer,
)


# Admin serializers moved to serializers.py

class AdminCompanySerializer(drf_serializers.ModelSerializer):
    """Company serializer for admin — all fields writable except computed ones."""
    is_currently_active = drf_serializers.ReadOnlyField()
    owner_name = drf_serializers.SerializerMethodField()
    owner_username = drf_serializers.SerializerMethodField()
    product_count = drf_serializers.IntegerField(read_only=True, default=0)
    logo = drf_serializers.SerializerMethodField()
    plan_name = drf_serializers.CharField(source='plan.name', read_only=True)
    plan_price = drf_serializers.IntegerField(source='plan.price', read_only=True)
    pending_payment_id = drf_serializers.SerializerMethodField()

    def get_pending_payment_id(self, obj):
        from ..models import Payment
        pending = Payment.objects.filter(company=obj, status='pending').order_by('-created_at').first()
        return pending.id if pending else None

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
        try:
            import datetime as dt
            now = timezone.now()
            thirty_days_ago = now - dt.timedelta(days=30)
            sixty_days_ago = now - dt.timedelta(days=60)

            # AI Status breakdown
            ai_stats = Product.objects.values('ai_status').annotate(count=Count('ai_status'))
            ai_breakdown = {s: 0 for s, _ in Product._meta.get_field('ai_status').choices}
            for item in ai_stats:
                ai_breakdown[item['ai_status']] = item['count']

            # Calculate success rate
            total_ai_processed = ai_breakdown.get('completed', 0) + ai_breakdown.get('error', 0)
            success_rate = round((ai_breakdown.get('completed', 0) / total_ai_processed * 100), 1) if total_ai_processed > 0 else 100

            # Recent activity
            recent_products = Product.objects.select_related('category').order_by('-id')[:5]
            recent_ai = AIResult.objects.select_related('user', 'product').order_by('-created_at')[:5]
            recent_leads = LeadRequest.objects.select_related('user', 'product').order_by('-created_at')[:5]

            # ── Growth (month-over-month) ───────────────────────────
            def calc_growth(model, date_field='created_at'):
                this_month = model.objects.filter(**{f'{date_field}__gte': thirty_days_ago}).count()
                last_month = model.objects.filter(
                    **{f'{date_field}__gte': sixty_days_ago, f'{date_field}__lt': thirty_days_ago}
                ).count()
                if last_month == 0:
                    return 100 if this_month > 0 else 0
                return round((this_month - last_month) / last_month * 100, 1)

            growth = {
                'products': calc_growth(Product),
                'companies': calc_growth(Company),
                'users': calc_growth(TelegramUser),
                'leads': calc_growth(LeadRequest),
                'ai_results': calc_growth(AIResult),
            }

            # ── Billing (SaaS cost tracking) ────────────────────────
            from ..models import SystemBilling
            sys_billing = SystemBilling.get_solo()

            ai_total_requests = AIResult.objects.count()
            ai_this_month = AIResult.objects.filter(created_at__gte=thirty_days_ago).count()
            ai_today = AIResult.objects.filter(created_at__date=now.date()).count()
            
            cost_per_req_usd = float(sys_billing.ai_cost_per_request)
            usd_to_uzs = float(sys_billing.usd_to_uzs_rate)
            cost_per_req_uzs = round(cost_per_req_usd * usd_to_uzs, 2)
            monthly_budget_uzs = float(sys_billing.ai_monthly_budget_uzs)

            server_due_date = sys_billing.server_due_date
            server_cost = sys_billing.server_cost
            server_note = sys_billing.server_note
            server_days_left = None
            if server_due_date:
                server_days_left = (server_due_date - now.date()).days

            ai_due_date = sys_billing.ai_due_date
            ai_days_left = None
            if ai_due_date:
                ai_days_left = (ai_due_date - now.date()).days

            billing = {
                'server_due_date': server_due_date.isoformat() if server_due_date else None,
                'server_cost': server_cost,
                'server_note': server_note,
                'server_days_left': server_days_left,
                'ai_due_date': ai_due_date.isoformat() if ai_due_date else None,
                'ai_cost_per_request_usd': cost_per_req_usd,
                'ai_cost_per_request_uzs': cost_per_req_uzs,
                'usd_to_uzs_rate': usd_to_uzs,
                'ai_monthly_budget_uzs': monthly_budget_uzs,
                'ai_total_requests': ai_total_requests,
                'ai_cost_total_uzs': round(ai_total_requests * cost_per_req_uzs, 2),
                'ai_cost_this_month_uzs': round(ai_this_month * cost_per_req_uzs, 2),
                'ai_cost_today_uzs': round(ai_today * cost_per_req_uzs, 2),
                'ai_days_left': ai_days_left,
            }

            # ── Financials (Revenue & Profit) ───────────────────────
            approved_payments = Payment.objects.filter(status='approved')
            income_today = approved_payments.filter(created_at__date=now.date()).aggregate(Sum('amount'))['amount__sum'] or 0
            income_month = approved_payments.filter(created_at__month=now.month, created_at__year=now.year).aggregate(Sum('amount'))['amount__sum'] or 0
            income_total = approved_payments.aggregate(Sum('amount'))['amount__sum'] or 0

            exp_ai_today = billing['ai_cost_today_uzs']
            exp_ai_month = billing['ai_cost_this_month_uzs']
            exp_ai_total = billing['ai_cost_total_uzs']
            
            exp_server_month = server_cost
            exp_server_today = round(server_cost / 30, 2)
            
            profit_today = float(income_today) - float(exp_ai_today) - float(exp_server_today)
            profit_month = float(income_month) - float(exp_ai_month) - float(exp_server_month)
            profit_total = float(income_total) - float(exp_ai_total)

            # Top Companies
            top_companies = Company.objects.annotate(
                total_paid=Sum('payments__amount', filter=Q(payments__status='approved'))
            ).filter(total_paid__gt=0).order_by('-total_paid')[:5]
            
            top_companies_data = [
                {
                    'id': c.id,
                    'name': c.name,
                    'total_paid': float(c.total_paid or 0),
                    'logo': c.logo.url if c.logo else None
                } for c in top_companies
            ]

            # 7-day Chart Data
            chart_data = []
            for i in range(6, -1, -1):
                day = now - timezone.timedelta(days=i)
                day_income = approved_payments.filter(created_at__date=day.date()).aggregate(Sum('amount'))['amount__sum'] or 0
                chart_data.append({
                    'date': day.strftime('%d.%m'),
                    'income': float(day_income)
                })

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
                'monetization': {
                    'pending_payment': Company.objects.filter(status='pending_payment').count(),
                    'waiting_confirmation': Company.objects.filter(status='waiting_confirmation').count(),
                    'active': Company.objects.filter(Q(status='active') | Q(is_vip=True)).count(),
                    'expired': Company.objects.filter(status='expired', is_vip=False).count(),
                },
                'financials': {
                    'income_today': float(income_today),
                    'income_month': float(income_month),
                    'income_total': float(income_total),
                    'profit_today': round(profit_today, 2),
                    'profit_month': round(profit_month, 2),
                    'profit_total': round(profit_total, 2),
                    'top_companies': top_companies_data,
                    'chart_data': chart_data,
                },
                'growth': growth,
                'billing': billing,
                'today_performance': {
                    'leads': LeadRequest.objects.filter(created_at__date=now.date()).count(),
                    'converted': LeadRequest.objects.filter(created_at__date=now.date(), status='converted').count(),
                    'conversion_rate': round(
                        (LeadRequest.objects.filter(created_at__date=now.date(), status='converted').count() /
                         max(LeadRequest.objects.filter(created_at__date=now.date()).count(), 1) * 100), 1
                    )
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
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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

    def perform_create(self, serializer):
        # Always mark admin-created products as active
        serializer.save(is_active=True)

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
    http_method_names = ['get', 'post', 'head', 'options']  # allow GET and custom POST actions

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
        return Response(AdminCompanySerializer(company, context={'request': request}).data)

    @action(detail=True, methods=['post'], url_path='toggle-vip')
    def toggle_vip(self, request, pk=None):
        company = self.get_object()
        company.is_vip = not company.is_vip
        company.save(update_fields=['is_vip'])
        return Response(AdminCompanySerializer(company, context={'request': request}).data)

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
        return Response(AdminCompanySerializer(company, context={'request': request}).data)

    @action(detail=True, methods=['post'], url_path='accept-payment')
    def accept_payment(self, request, pk=None):
        company = self.get_object()
        duration = 30
        if company.plan:
            duration = company.plan.duration_days
        
        now = timezone.now()
        current_deadline = company.subscription_deadline
        base = current_deadline if current_deadline and current_deadline > now else now
        new_deadline = base + datetime.timedelta(days=duration)
        
        company.subscription_deadline = new_deadline
        company.is_active = True
        company.save(update_fields=["subscription_deadline", "is_active"])
        
        return Response(AdminCompanySerializer(company, context={'request': request}).data)



# ── Promotions (on-sale products) ───────────────────────────
class AdminPromotionViewSet(viewsets.ModelViewSet):
    serializer_class = ProductSerializer
    permission_classes = [IsAdminUser]
    http_method_names = ['get', 'post', 'head', 'options', 'patch', 'delete']

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

    @action(detail=True, methods=['post'], url_path='broadcast')
    def broadcast(self, request, pk=None):
        """Send this promotion to ALL Telegram users as a broadcast."""
        from shop.notifications import NotificationService
        product = self.get_object()
        if not product.is_on_sale:
            return Response(
                {"detail": "Mahsulot aksiyada emas."},
                status=status.HTTP_400_BAD_REQUEST
            )
        sent, failed = NotificationService.broadcast_promotion(product)
        return Response({
            "detail": f"Broadcast yakunlandi: {sent} ta yuborildi, {failed} ta xato.",
            "sent": sent,
            "failed": failed,
        })


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
    http_method_names = ['get', 'head', 'options', 'patch', 'delete', 'post']

    def get_queryset(self):
        qs = LeadRequest.objects.select_related('user', 'product', 'company').order_by('-created_at')
        status = self.request.query_params.get('status')
        if status and status != 'all':
            qs = qs.filter(status=status)
        return qs

    @action(detail=True, methods=['post'], url_path='set-status')
    def set_status(self, request, pk=None):
        lead = self.get_object()
        new_status = request.data.get('status')
        if new_status in dict(LeadRequest.STATUS_CHOICES):
            lead.status = new_status
            # Keep is_processed in sync for compatibility
            lead.is_processed = (new_status in ['converted', 'closed', 'rejected'])
            lead.save(update_fields=['status', 'is_processed'])
            return Response(AdminLeadRequestSerializer(lead, context={'request': request}).data)
        return Response({'error': 'Invalid status'}, status=400)

    @action(detail=True, methods=['post'], url_path='toggle-processed')
    def toggle_processed(self, request, pk=None):
        lead = self.get_object()
        lead.is_processed = not lead.is_processed
        if lead.is_processed:
            lead.status = 'closed'
        else:
            lead.status = 'new'
        lead.save(update_fields=['is_processed', 'status'])
        return Response(AdminLeadRequestSerializer(lead, context={'request': request}).data)


# ── AI Results ──────────────────────────────────────────────
class AdminAIResultViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = AdminAIResultSerializer
    permission_classes = [IsAdminUser]

    def get_queryset(self):
        return AIResult.objects.select_related('user', 'product').order_by('-created_at')
# ── AI Tests (Admin Lab) ────────────────────────────────────
class AdminAITestViewSet(viewsets.ModelViewSet):
    queryset = AITest.objects.all()
    serializer_class = AITestSerializer
    permission_classes = [IsAdminUser]

    @action(detail=True, methods=['post'])
    def run_test(self, request, pk=None):
        import os
        import uuid
        from django.conf import settings

        test_obj = self.get_object()
        product = test_obj.door
        
        # Determine paths
        request_id = f"test_{test_obj.id}_{uuid.uuid4().hex[:8]}"
        result_dir = os.path.join(settings.MEDIA_ROOT, 'ai_tests', 'results')
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"{request_id}.png")
        
        try:
            # We need the physical path of the room image
            if not test_obj.room_image:
                 return Response({'error': 'Xonaning rasmi yuklanmagan!'}, status=status.HTTP_400_BAD_REQUEST)
            room_path = test_obj.room_image.path

            from ..services import AIService
            # Run the locked-scene hybrid pipeline.
            # SAM is optional inside the service; if unavailable it falls back to YOLO/OpenCV.
            AIService.generate_room_preview(
                product, 
                room_path, 
                result_path
            )
            
            # Save the result image relative path
            image_rel_path = f"ai_tests/results/{os.path.basename(result_path)}"
            test_obj.result_image = image_rel_path
            
            # Upload to Telegram as permanent storage
            from ..notifications import NotificationService
            import datetime
            caption = (
                f"🧪 AI Test Result\n"
                f"🕒 Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                f"🚪 Product: {product.name}\n\n"
                f"#tanlaai #aitest"
            )
            file_id = NotificationService.upload_photo_to_telegram(result_path, caption)
            if file_id:
                test_obj.telegram_file_id = file_id

            # Load metadata if exists
            from ..ai_utils import load_visualization_metadata
            try:
                metadata = load_visualization_metadata(result_path) or {}
                if test_obj.prompt:
                    metadata.setdefault('pipeline', {})
                    metadata['pipeline']['tester_note'] = test_obj.prompt
                test_obj.metadata = metadata
            except:
                pass
                
            test_obj.save()
            return Response(AITestSerializer(test_obj, context={'request': request}).data)
            
        except Exception as e:
            import traceback
            return Response({
                'error': str(e),
                'traceback': traceback.format_exc()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ── Payment moderation (admin) ───────────────────────────────
class AdminPaymentViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Admin panel endpoints for reviewing company payment submissions.

    - GET /api/admin/payments/            — list all (filter by ?status=pending)
    - GET /api/admin/payments/{id}/       — detail
    - POST /api/admin/payments/{id}/approve/  — extend subscription + reactivate products
    - POST /api/admin/payments/{id}/reject/   — mark rejected (body: {reason})
    """
    serializer_class = PaymentSerializer
    permission_classes = [IsAdminUser]

    def get_queryset(self):
        qs = Payment.objects.select_related(
            "company", "company__user", "reviewed_by"
        )
        status_filter = self.request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)
        return qs

    @action(detail=True, methods=["post"], permission_classes=[IsAdminUser])
    def approve(self, request, pk=None):
        from ..payment_service import PaymentService
        payment = self.get_object()
        
        tg_user = TelegramUser.objects.filter(
            telegram_id=getattr(request.user, "id", None)
        ).first()
        
        success, message = PaymentService.approve_payment(payment, reviewed_by_tg_user=tg_user)
        if not success:
            return Response({"detail": message}, status=status.HTTP_400_BAD_REQUEST)

        return Response(PaymentSerializer(payment, context={"request": request}).data)

    @action(detail=True, methods=["post"], permission_classes=[IsAdminUser])
    def reject(self, request, pk=None):
        from ..payment_service import PaymentService
        payment = self.get_object()
        
        reason = (request.data.get("reason") or "").strip()
        if not reason:
            return Response(
                {"reason": "Rad etish sababi kiritilishi shart."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        tg_user = TelegramUser.objects.filter(
            telegram_id=getattr(request.user, "id", None)
        ).first()

        success, message = PaymentService.reject_payment(payment, reason, reviewed_by_tg_user=tg_user)
        if not success:
            return Response({"detail": message}, status=status.HTTP_400_BAD_REQUEST)

        return Response(PaymentSerializer(payment, context={"request": request}).data)
