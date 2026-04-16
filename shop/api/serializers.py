from rest_framework import serializers
from ..models import (
    Category, TelegramUser, Company, Product, 
    AIResult, HomeBanner, Wishlist, LeadRequest, Subscription, AITest, SharedDesign
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
        if url and not url.startswith('/') and not url.startswith('http'):
            url = f"/media/{url}"
        
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(url)
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

class ProductSerializer(serializers.ModelSerializer):
    category_name = serializers.ReadOnlyField(source='category.name')
    company_details = CompanySerializer(source='company', read_only=True)
    owner_details = TelegramUserSerializer(source='owner', read_only=True)
    is_wishlisted = serializers.SerializerMethodField()
    
    # Absolute URL fields (Writable)
    image = AbsoluteImageField(required=False, allow_null=True)
    original_image = AbsoluteImageField(required=False, allow_null=True)
    image_no_bg = AbsoluteImageField(required=False, allow_null=True)
    
    class Meta:
        model = Product
        fields = '__all__'
        read_only_fields = ['owner', 'company', 'category_name', 'company_details', 'owner_details', 'is_wishlisted', 'ai_status']

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
        read_only_fields = ['user', 'company', 'created_at', 'product_name', 'product_image']

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
