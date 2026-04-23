from pathlib import Path
from unittest.mock import patch

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings

from shop.models import LeadRequest, Wishlist
from shop.tests.factories import CategoryFactory, CompanyFactory, ProductFactory, TelegramUserFactory


@override_settings(ALLOWED_HOSTS=['testserver', 'localhost', '127.0.0.1'])
class AppFlowTests(TestCase):
    def setUp(self):
        self.client = Client()

    def login_as_telegram_user(self, user):
        session = self.client.session
        session['tg_user_id'] = user.id
        session.save()

    def ensure_spa_index(self):
        index_path = Path(settings.BASE_DIR) / 'static' / 'react' / 'index.html'
        if index_path.exists():
            return

        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            '<!doctype html><html><head><link rel="stylesheet" href="/static/react/assets/test.css"></head>'
            '<body><div id="root"></div><script src="/static/react/assets/test.js"></script></body></html>',
            encoding='utf-8',
        )
        self.addCleanup(index_path.unlink)

    def test_spa_routes_return_react_index(self):
        self.ensure_spa_index()
        for route in ['/', '/search', '/leaders', '/discounts', '/profile', '/wishlist', '/product/999', '/company/999']:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, 'id="root"', html=False)
            self.assertContains(response, '/static/react/assets/', html=False)

    def test_admin_dashboard_redirects_to_login(self):
        login_response = self.client.get('/login/')
        dashboard_response = self.client.get('/admin-dashboard/')

        self.assertEqual(login_response.status_code, 200)
        self.assertEqual(dashboard_response.status_code, 302)
        self.assertIn('/login/?next=/admin-dashboard/', dashboard_response['Location'])

    def test_creator_can_upsert_company_and_create_product(self):
        creator = TelegramUserFactory()
        category = CategoryFactory(name='Premium Eshiklar')
        self.login_as_telegram_user(creator)

        create_company = self.client.post('/api/v1/companies/my/', {
            'name': 'Artisan Doors',
            'description': 'Premium eshiklar ishlab chiqaruvchisi',
            'location': 'Tashkent',
            'telegram_link': '@artisan',
        })
        self.assertEqual(create_company.status_code, 201)
        self.assertEqual(create_company.json()['name'], 'Artisan Doors')

        update_company = self.client.patch(
            '/api/v1/companies/my/',
            data={'description': 'Yangilangan tavsif'},
            content_type='application/json',
        )
        self.assertEqual(update_company.status_code, 200)
        self.assertEqual(update_company.json()['description'], 'Yangilangan tavsif')

        create_product = self.client.post('/api/v1/products/', {
            'name': 'Akustik eshik',
            'description': 'Issiqlik va ovoz izolyatsiyasi kuchli',
            'category': category.id,
            'price': '1800000',
            'height': '200',
            'width': '80',
            'is_featured': 'true',
            'is_on_sale': 'false',
        })
        self.assertEqual(create_product.status_code, 201)
        self.assertEqual(create_product.json()['company_details']['name'], 'Artisan Doors')

        my_products = self.client.get('/api/v1/products/my/')
        self.assertEqual(my_products.status_code, 200)
        self.assertEqual(len(my_products.json()), 1)
        self.assertEqual(my_products.json()[0]['name'], 'Akustik eshik')

    def test_wishlist_and_lead_creation_flow(self):
        product = ProductFactory()
        viewer = TelegramUserFactory()
        self.login_as_telegram_user(viewer)

        toggle_wishlist = self.client.post(f'/api/v1/products/{product.id}/toggle_wishlist/')
        self.assertEqual(toggle_wishlist.status_code, 200)
        self.assertEqual(toggle_wishlist.json()['status'], 'added')
        self.assertTrue(Wishlist.objects.filter(user=viewer, product=product).exists())

        wishlist_response = self.client.get('/api/v1/wishlist/')
        self.assertEqual(wishlist_response.status_code, 200)
        self.assertEqual(wishlist_response.json()['count'], 1)

        lead_response = self.client.post('/api/v1/leads/', {
            'product': product.id,
            'lead_type': 'measurement',
            'message': 'Ertaga o\'lchov olish mumkinmi?',
            'phone': '+998901234567',
            'address_text': 'Toshkent, Chilonzor tumani',
        })
        self.assertEqual(lead_response.status_code, 201)
        self.assertTrue(LeadRequest.objects.filter(user=viewer, product=product, company=product.company).exists())

    def test_ai_generate_prechecks_work_without_running_external_ai(self):
        viewer = TelegramUserFactory()
        company = CompanyFactory()
        product = ProductFactory(company=company, category=CategoryFactory(name='Premium Eshiklar'), height='200', width='80')

        unauthenticated = self.client.post(f'/api/v1/products/{product.id}/ai-generate/')
        self.assertEqual(unauthenticated.status_code, 401)

        self.login_as_telegram_user(viewer)
        missing_photo = self.client.post(f'/api/v1/products/{product.id}/ai-generate/', {
            'height': '200',
            'width': '80',
        })
        self.assertEqual(missing_photo.status_code, 400)
        self.assertEqual(missing_photo.json()['code'], 'no_photo')

        mismatch = self.client.post(
            f'/api/v1/products/{product.id}/ai-generate/',
            {
                'height': '220',
                'width': '90',
                'room_photo': SimpleUploadedFile('room.png', b'fake-image', content_type='image/png'),
            },
        )
        self.assertEqual(mismatch.status_code, 400)
        self.assertEqual(mismatch.json()['code'], 'dimension_mismatch')

    def test_ai_generate_stores_request_state_in_session(self):
        viewer = TelegramUserFactory()
        company = CompanyFactory()
        product = ProductFactory(
            company=company,
            category=CategoryFactory(name='Premium Eshiklar'),
            ai_status='completed',
        )
        self.login_as_telegram_user(viewer)

        with patch('shop.api.views.ai_executor.submit') as submit_mock:
            response = self.client.post(
                f'/api/v1/products/{product.id}/ai-generate/',
                {
                    'room_photo': SimpleUploadedFile('room.png', b'fake-image', content_type='image/png'),
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload['status'], 'ok')
        self.assertTrue(payload['request_id'])

        session = self.client.session
        job_state = session[f'ai_gen_{product.id}']
        self.assertEqual(job_state['status'], 'running')
        self.assertEqual(job_state['request_id'], payload['request_id'])
        submit_mock.assert_called_once()

    def test_ai_generate_result_prefers_session_error_message(self):
        viewer = TelegramUserFactory()
        product = ProductFactory(ai_status='completed')
        self.login_as_telegram_user(viewer)

        session = self.client.session
        session[f'ai_gen_{product.id}'] = {
            'status': 'error',
            'request_id': 'req-123',
            'error_msg': 'Door asset missing',
        }
        session.save()

        response = self.client.get(
            f'/api/v1/products/{product.id}/ai-generate/result/',
            {'request_id': 'req-123'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'error')
        self.assertEqual(response.json()['message'], 'Door asset missing')
