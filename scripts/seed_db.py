import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.models import Category, Product

def seed():
    # Categories
    apparel, _ = Category.objects.get_or_create(name='Apparel', icon='checkroom')
    luxury, _ = Category.objects.get_or_create(name='Luxury', icon='diamond')
    accessories, _ = Category.objects.get_or_create(name='Accessories', icon='watch')

    # Products
    products_data = [
        {
            'name': 'Heritage Trench Coat',
            'description': 'A classic tan double-breasted trench coat draped elegantly for a timeless look.',
            'price': 840.00,
            'image_url': 'https://lh3.googleusercontent.com/aida-public/AB6AXuBARqWJLO6bWJ5kLyw3CCJTfzhmR3eHADguhI-1OmwPi1nuFov9XEvTwv0g0j3u2fOs1CMr4W4cUkQFQ2g2VLYjiQCp2eMOrLmMPxKj3yf17tToSa8PVKwMYK9nxHf6-bWi1naAF0IRAiwR8viveCwpLcznZGi0_09MmEUNCbz-FUahu8O0OwGxKnK-O_vzMEWO_FZL-h4XCNgiedOa6I9EBdNavOfRotSxRHG20JOlIGQs0ucYolj1JQlNj1Fi2gxiifERL6ZUtuRM',
            'category': apparel,
            'is_featured': True
        },
        {
            'name': 'Crimson Aero Sneaker',
            'description': 'Vibrant red luxury leather sneakers isolated on a minimalist gray background.',
            'price': 210.00,
            'image_url': 'https://lh3.googleusercontent.com/aida-public/AB6AXuBeKbUv1F_vEFYkzBROdVWnJQ5jsaiqAgMc4aG2f_NzXkJ0uQJPqqktA26BG2xrW5g_9BlqkUyBLnYKzw3eWfYWfCF1z9iZRCptBgjDwM8H5h90KprsN4cyiSMbA7B7prMaWvWtaj4YrKVAllLCX5CVshjMddrVABlhT7CFV1hu4LtaDbmU_glVDWHFkPDsRfDK1m8M1vnApwY1lq6keQnU8ZnsPPKcGxK2wo25baEa4zJ5nEKjLxMc3pMRIYyENJ_1HSNyIN2DwA8E',
            'category': apparel,
            'is_featured': True
        },
        {
            'name': 'Studio Pro Wireless',
            'description': 'High-fidelity gold and black studio headphones with dramatic rim lighting.',
            'price': 350.00,
            'image_url': 'https://lh3.googleusercontent.com/aida-public/AB6AXuDcsBnFeogVQ-6Apf3zDUG1n1e4BDUTKSFcNrnRmTqKT_-KVG9oN1TdF62epX4_qG1Fr6-BjDY3hICggW2z6xy0UqbNcsw2XWwKvSqLHSjtm-_6Xb35AXCWDwhDxgeEVmZyNl7ggnQG7yhLwTYgkOgm_vKCZgpdoN0dHp8D2wIDmHhk9a4kTUKEPxPqZRDY7GsvRApGmCUj7mFLGXJSyTzFgbo5-JwHJrTAVoNypHyzLFTWnCmunE_sbZIRVz8KumKAsnX4UG2bZaiM',
            'category': accessories,
            'is_featured': True
        },
        {
            'name': 'Chronos Series 01',
            'description': 'Luxury silver mechanical watch with intricate dial details on a dark velvet cushion.',
            'price': 1200.00,
            'image_url': 'https://lh3.googleusercontent.com/aida-public/AB6AXuCAHhpFtUhfoOTY9xFaQZBvJfkAQt7_gBNfmEETE8Dcywlktdvj9a6DaM5KK5A50VOYIGEHfcVfWP2Jp1ycmSTXIFSxa2z66_pDIzlZow5eEOq31zPmX7cxZJKAGcwqvLjiUJEgQKPdI66jx-UwWBaqZGSeH8gHrSAQfomOYA1SknkSCTYW1Y872bI5OAElre0cmyWMD2jovaWNkmWQhRN-iCPLNpgcP3xnTT15x7QjUbKFItfPB4xGCeTbPSBiRqckyItqPzosxAie',
            'category': luxury,
            'is_featured': True
        },
        {
            'name': 'Chronos Series v2',
            'description': 'Refined engineering with titanium silver casing and ocean blue silicone strap.',
            'price': 849.00,
            'image_url': 'https://lh3.googleusercontent.com/aida-public/AB6AXuAzBQvYMnf_umHYz-ThUwW62IZUdfxtJxoi4bLrhHDdfSLuuqvYhJdpOE0sNhlVayRW15vNKk65rWSsapB8tiRO1e_83suDsGVZJjVwS6KEG-vA3w1sT7xR3ixYks58-pF81gOXyl90RaOGb4GyU0viSWEEFfOS3GtHAxwh1QtyzXeqSr-fVNfjh1sGXX7YJ0P4_RS58fLsRJzen-fBgt5d6U_OgFC7Sq1cCMS5q_3JgSqrxRR_5f8zUFxqK2lsbxYRF9UoWFs21h-K',
            'category': luxury,
            'is_featured': False
        }
    ]

    for p_data in products_data:
        Product.objects.get_or_create(
            name=p_data['name'],
            defaults={
                'description': p_data['description'],
                'price': p_data['price'],
                'image_url': p_data['image_url'],
                'category': p_data['category'],
                'is_featured': p_data['is_featured']
            }
        )

    print("Database seeded successfully!")

if __name__ == '__main__':
    seed()
