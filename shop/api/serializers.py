from django.conf import settings
from rest_framework import serializers
from ..models import (
    Category, TelegramUser, Company, Product, ProductImage,
    AIResult, HomeBanner, Wishlist, LeadRequest, Subscription, AITest, SharedDesign,
    Payment,
)

class AbsoluteImageField(serializers.ImageField):
    """
    Custom ImageField that returns an absolute URL in the representation
    while remaining writable for create/update operations.
    """
    def to_representation(self, value):
        if not value:
            return None
        url = value.url
        # 1. Normalize relative paths
        if url and not url.startswith('/') and not url.startswith('http'):
            url = f"/media/{url}"
        
        # 2. Try request context
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(url)
        
        # 3. Fallback to settings.BACKEND_URL
        if url.startswith('/'):
            backend_url = getattr(settings, 'BACKEND_URL', '').rstrip('/')
            return f"{backend_url}{url}"
            
        return url

class CategorySerializer(serializers.ModelSerializer):
    product_count = serializers.IntegerField(read_only=True, default=0)
    icon = AbsoluteImageField(required=False, allow_null=True)

    class Meta:
        model = Category
        fields = ['id', 'name', 'icon', 'product_count']

class TelegramUserSerializer(serializers.ModelSerializer):
    has_company = serializers.SerializerMethodField()

    class Meta:
        model = TelegramUser
        fields = ['id', 'telegram_id', 'first_name', 'last_name', 'username', 'role', 'photo_url', 'has_company', 'created_at']

    def get_has_company(self, obj):
        return hasattr(obj, 'company')

class CompanySerializer(serializers.ModelSerializer):
    is_currently_active = serializers.ReadOnlyField()
    logo = AbsoluteImageField(required=False, allow_null=True)
    
    class Meta:
        model = Company
        fields = '__all__'
        read_only_fields = ['user', 'is_active', 'subscription_deadline', 'created_at', 'is_currently_active']

class ProductImageSerializer(serializers.ModelSerializer):
    image = AbsoluteImageField()

    class Meta:
        model = ProductImage
        fields = ['id', 'image', 'is_main', 'order']
        read_only_fields = ['id']


class ProductSerializer(serializers.ModelSerializer):
    category_name = serializers.ReadOnlyField(source='category.name')
    company_details = CompanySerializer(source='company', read_only=True)
    owner_details = TelegramUserSerializer(source='owner', read_only=True)
    is_wishlisted = serializers.SerializerMethodField()

    # Absolute URL fields (Writable)
    image = AbsoluteImageField(required=False, allow_null=True)
    original_image = AbsoluteImageField(required=False, allow_null=True)
    image_no_bg = AbsoluteImageField(required=False, allow_null=True)

    # Nested gallery (read-only here — write-side is handled in ProductViewSet)
    images = ProductImageSerializer(many=True, read_only=True)

    class Meta:
        model = Product
        fields = '__all__'
        read_only_fields = [
            'owner', 'company', 'category_name', 'company_details',
            'owner_details', 'is_wishlisted', 'ai_status', 'images',
        ]

    def get_is_wishlisted(self, obj):
        request = self.context.get('request')
        if not request:
            return False
        tg_user_id = request.session.get('tg_user_id')
        if not tg_user_id:
            return False
        return Wishlist.objects.filter(product=obj, user_id=tg_user_id).exists()

class AIResultSerializer(serializers.ModelSerializer):
    product_details = ProductSerializer(source='product', read_only=True)
    image = AbsoluteImageField(read_only=True)
    input_image = AbsoluteImageField(read_only=True)

    class Meta:
        model = AIResult
        fields = '__all__'

class HomeBannerSerializer(serializers.ModelSerializer):
    image = AbsoluteImageField(required=False, allow_null=True)

    class Meta:
        model = HomeBanner
        fields = '__all__'

class WishlistSerializer(serializers.ModelSerializer):
    product_details = ProductSerializer(source='product', read_only=True)
    
    class Meta:
        model = Wishlist
        fields = '__all__'

class LeadRequestSerializer(serializers.ModelSerializer):
    company_name = serializers.CharField(source='company.name', read_only=True)
    user_name = serializers.CharField(source='user.first_name', read_only=True)
    product_name = serializers.ReadOnlyField(source='product.name')
    product_image = AbsoluteImageField(source='product.image', read_only=True)
    ai_result_image = serializers.SerializerMethodField()
    ai_result_details = AIResultSerializer(source='ai_result', read_only=True)

    class Meta:
        model = LeadRequest
        fields = '__all__'
        # `calculated_price` is read-only from the API's POV:
        # server always recomputes it from product.price_per_m2 × (w*h/10000).
        read_only_fields = [
            'user', 'company', 'created_at',
            'product_name', 'product_image', 'calculated_price',
        ]

    # Require at least one form of address for checkout-style leads.
    # We DO NOT enforce this for `visualize` (AI auto-lead) or `telegram`
    # — those flows don't collect an address.
    # `direct` = customer orders the product directly without AI visualization.
    CHECKOUT_LEAD_TYPES = {'call', 'measurement', 'direct'}

    def validate(self, attrs):
        lead_type = attrs.get('lead_type')
        if lead_type in self.CHECKOUT_LEAD_TYPES:
            address_text = (attrs.get('address_text') or '').strip()
            lat = attrs.get('latitude')
            lng = attrs.get('longitude')
            has_coords = lat is not None and lng is not None
            if not address_text and not has_coords:
                raise serializers.ValidationError({
                    'address': "Manzil yoki lokatsiya kerak."
                })
        return attrs

    def get_ai_result_image(self, obj):
        if not obj.ai_result or not obj.ai_result.image:
            return None
        
        url = obj.ai_result.image.url
        if url and not url.startswith('/') and not url.startswith('http'):
            url = f"/media/{url}"
        
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(url)
        return url

# Admin aliases for backward compatibility
AdminLeadRequestSerializer = LeadRequestSerializer

class AdminAIResultSerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(source='user.first_name', read_only=True)
    product_name = serializers.CharField(source='product.name', read_only=True)
    image = AbsoluteImageField(read_only=True)

    class Meta:
        model = AIResult
        fields = '__all__'

class SubscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subscription
        fields = '__all__'


class PaymentSerializer(serializers.ModelSerializer):
    """
    Payment submission/read serializer.

    Owner-writable fields:  amount, months, screenshot, note
    Server-controlled:      status, rejection_reason, reviewed_at, reviewed_by,
                            company (set from request.user's company), created_at
    """
    company_name = serializers.CharField(source='company.name', read_only=True)
    screenshot = AbsoluteImageField()
    reviewed_by_name = serializers.CharField(
        source='reviewed_by.first_name', read_only=True, default=None
    )
    status_display = serializers.CharField(source='get_status_display', read_only=True)

    class Meta:
        model = Payment
        fields = [
            'id', 'company', 'company_name', 'amount', 'months',
            'screenshot', 'note',
            'status', 'status_display', 'rejection_reason',
            'created_at', 'reviewed_at', 'reviewed_by', 'reviewed_by_name',
        ]
        read_only_fields = [
            'id', 'company', 'company_name',
            'status', 'status_display', 'rejection_reason',
            'created_at', 'reviewed_at', 'reviewed_by', 'reviewed_by_name',
        ]

    def validate_amount(self, value):
        if value <= 0:
            raise serializers.ValidationError("Summa 0 dan katta bo'lishi kerak.")
        return value

    def validate_months(self, value):
        # Sanity bound — we don't want a typo to extend a subscription by 9999 months.
        if value < 1 or value > 24:
            raise serializers.ValidationError("Oylar soni 1–24 oralig'ida bo'lishi kerak.")
        return value

class AITestSerializer(serializers.ModelSerializer):
    door_details = ProductSerializer(source='door', read_only=True)
    room_image = AbsoluteImageField(required=True)
    result_image = AbsoluteImageField(read_only=True)

    class Meta:
        model = AITest
        fields = '__all__'


class SharedDesignSerializer(serializers.ModelSerializer):
    product_details = ProductSerializer(source='product', read_only=True)
    image = AbsoluteImageField(required=True)
    original_image = AbsoluteImageField(required=False, allow_null=True)
    
    class Meta:
        model = SharedDesign
        fields = '__all__'
        read_only_fields = ['id', 'created_at']
