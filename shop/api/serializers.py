from rest_framework import serializers
from ..models import (
    Category, TelegramUser, Company, Product, 
    AIResult, HomeBanner, Wishlist, LeadRequest, Subscription
)

class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = '__all__'

class TelegramUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = TelegramUser
        fields = '__all__'

class CompanySerializer(serializers.ModelSerializer):
    is_currently_active = serializers.ReadOnlyField()
    
    class Meta:
        model = Company
        fields = '__all__'
        read_only_fields = ['user', 'is_active', 'subscription_deadline', 'created_at', 'is_currently_active']

class ProductSerializer(serializers.ModelSerializer):
    category_name = serializers.ReadOnlyField(source='category.name')
    company_details = CompanySerializer(source='company', read_only=True)
    owner_details = TelegramUserSerializer(source='owner', read_only=True)
    is_wishlisted = serializers.SerializerMethodField()
    
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

    class Meta:
        model = AIResult
        fields = '__all__'

class HomeBannerSerializer(serializers.ModelSerializer):
    class Meta:
        model = HomeBanner
        fields = '__all__'

class WishlistSerializer(serializers.ModelSerializer):
    product_details = ProductSerializer(source='product', read_only=True)
    
    class Meta:
        model = Wishlist
        fields = '__all__'

class LeadRequestSerializer(serializers.ModelSerializer):
    product_name = serializers.ReadOnlyField(source='product.name')
    user_details = TelegramUserSerializer(source='user', read_only=True)
    
    class Meta:
        model = LeadRequest
        fields = '__all__'
        read_only_fields = ['user', 'company', 'created_at', 'product_name', 'user_details']

class SubscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subscription
        fields = '__all__'
