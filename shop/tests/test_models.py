from django.test import TestCase
from shop.models import Wishlist, Subscription
from shop.tests.factories import TelegramUserFactory, ProductFactory, CompanyFactory

class ModelTests(TestCase):
    def test_wishlist_toggle(self):
        user = TelegramUserFactory()
        product = ProductFactory()
        
        # Create wishlist item
        Wishlist.objects.create(user=user, product=product)
        self.assertEqual(Wishlist.objects.filter(user=user, product=product).count(), 1)
        
        # Delete (toggle)
        Wishlist.objects.filter(user=user, product=product).delete()
        self.assertEqual(Wishlist.objects.filter(user=user, product=product).count(), 0)

    def test_subscription_creation_and_limits(self):
        company = CompanyFactory()
        # Subscription is usually created in views, manually testing here
        subscription, created = Subscription.objects.get_or_create(company=company)
        
        self.assertEqual(subscription.plan, 'free')
        self.assertEqual(subscription.max_products, 10)
        self.assertEqual(subscription.ai_generations_limit, 50)
        
        # Update plan
        subscription.plan = 'pro'
        subscription.max_products = 100
        subscription.save()
        
        self.assertEqual(Subscription.objects.get(company=company).plan, 'pro')
        self.assertEqual(Subscription.objects.get(company=company).max_products, 100)

