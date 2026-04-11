from rest_framework import serializers
from ..models import (
    Category, TelegramUser, Company, Product, 
    AIResult, HomeBanner, Wishlist, LeadRequest, Subscription
)

class CategorySerializer(serializers.ModelSerializer):
    product_count = serializers.IntegerField(read_only=True, default=0)
    icon = serializers.SerializerMethodField()

    class Meta:
        model = Category
        fields = ['id', 'name', 'icon', 'product_count']

    def get_icon(self, obj):
        if not obj.icon:
            return None
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(obj.icon.url)
        return obj.icon.url

class TelegramUserSerializer(serializers.ModelSerializer):
    has_company = serializers.SerializerMethodField()

    class Meta:
        model = TelegramUser
        fields = ['id', 'telegram_id', 'first_name', 'last_name', 'username', 'role', 'photo_url', 'has_company']

    def get_has_company(self, obj):
        return hasattr(obj, 'company')

class CompanySerializer(serializers.ModelSerializer):
    is_currently_active = serializers.ReadOnlyField()
    logo = serializers.SerializerMethodField()
    
    class Meta:
        model = Company
        fields = '__all__'
        read_only_fields = ['user', 'is_active', 'subscription_deadline', 'created_at', 'is_currently_active']

    def get_logo(self, obj):
        if not obj.logo:
            return None
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(obj.logo.url)
        return obj.logo.url

class ProductSerializer(serializers.ModelSerializer):
    category_name = serializers.ReadOnlyField(source='category.name')
    company_details = CompanySerializer(source='company', read_only=True)
    owner_details = TelegramUserSerializer(source='owner', read_only=True)
    is_wishlisted = serializers.SerializerMethodField()
    
    # Absolute URL fields
    image = serializers.SerializerMethodField()
    original_image = serializers.SerializerMethodField()
    image_no_bg = serializers.SerializerMethodField()
    
    class Meta:
        model = Product
        fields = '__all__'
        read_only_fields = ['owner', 'company', 'category_name', 'company_details', 'owner_details', 'is_wishlisted', 'ai_status']
        
    def _get_abs_url(self, field):
        if not field:
            return None
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(field.url)
        return field.url

    def get_image(self, obj): return self._get_abs_url(obj.image)
    def get_original_image(self, obj): return self._get_abs_url(obj.original_image)
    def get_image_no_bg(self, obj): return self._get_abs_url(obj.image_no_bg)

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
    image = serializers.SerializerMethodField()
    input_image = serializers.SerializerMethodField()

    class Meta:
        model = AIResult
        fields = '__all__'

    def _get_abs_url(self, field):
        if not field:
            return None
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(field.url)
        return field.url

    def get_image(self, obj): return self._get_abs_url(obj.image)
    def get_input_image(self, obj): return self._get_abs_url(obj.input_image)

class HomeBannerSerializer(serializers.ModelSerializer):
    image = serializers.SerializerMethodField()

    class Meta:
        model = HomeBanner
        fields = '__all__'

    def get_image(self, obj):
        if not obj.image:
            return None
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(obj.image.url)
        return obj.image.url

class WishlistSerializer(serializers.ModelSerializer):
    product_details = ProductSerializer(source='product', read_only=True)
    
    class Meta:
        model = Wishlist
        fields = '__all__'

class LeadRequestSerializer(serializers.ModelSerializer):
    company_name = serializers.CharField(source='company.name', read_only=True)
    user_name = serializers.CharField(source='user.first_name', read_only=True)
    product_name = serializers.ReadOnlyField(source='product.name')

    class Meta:
        model = LeadRequest
        fields = '__all__'
        read_only_fields = ['user', 'company', 'created_at', 'product_name']

# Admin aliases for backward compatibility
AdminLeadRequestSerializer = LeadRequestSerializer

class AdminAIResultSerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(source='user.first_name', read_only=True)
    product_name = serializers.CharField(source='product.name', read_only=True)
    image = serializers.SerializerMethodField()

    class Meta:
        model = AIResult
        fields = '__all__'
    
    def get_image(self, obj):
        if not obj.image:
            return None
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(obj.image.url)
        return obj.image.url

class SubscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subscription
        fields = '__all__'
