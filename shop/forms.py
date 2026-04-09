from django.db import models
from django import forms
from .models import Product, Category, Company, HomeBanner

class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'description', 'price', 'discount_price', 'is_on_sale', 'sale_end_date', 'image', 'category', 'height', 'width', 'price_per_m2', 'is_featured']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'Product name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none h-32',
                'placeholder': 'Product description'
            }),
            'price': forms.NumberInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'Total Price (0.00)'
            }),
            'category': forms.Select(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none'
            }),
            'height': forms.NumberInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'Height (cm)'
            }),
            'width': forms.NumberInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'Width (cm)'
            }),
            'price_per_m2': forms.NumberInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'Price per 1 m² (0.00)'
            }),
            'image': forms.FileInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none'
            }),
            'is_on_sale': forms.CheckboxInput(attrs={
                'class': 'w-5 h-5 rounded border-none bg-surface-container-low text-primary focus:ring-primary/10'
            }),
            'discount_price': forms.NumberInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'Sale price'
            }),
            'sale_end_date': forms.DateTimeInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'type': 'datetime-local'
            }),
            'is_featured': forms.CheckboxInput(attrs={
                'class': 'w-5 h-5 rounded border-none bg-surface-container-low text-primary focus:ring-primary/10'
            }),
        }
        
    pricing_type = forms.ChoiceField(
        choices=[('total', 'Total Price'), ('per_m2', 'Price per 1 m²')],
        widget=forms.HiddenInput(), # Actual UI will be a styled toggle
        initial='total',
        required=False
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            if self.instance.price_per_m2 and not self.instance.price:
                self.initial['pricing_type'] = 'per_m2'
            else:
                self.initial['pricing_type'] = 'total'

    def clean(self):
        cleaned_data = super().clean()
        pricing_type = cleaned_data.get("pricing_type")
        price = cleaned_data.get("price")
        price_per_m2 = cleaned_data.get("price_per_m2")

        if pricing_type == 'total':
            if not price:
                self.add_error('price', "Total Price is required when selected.")
            # Clear the other field to avoid confusion
            cleaned_data['price_per_m2'] = None
        elif pricing_type == 'per_m2':
            if not price_per_m2:
                self.add_error('price_per_m2', "Price per m² is required when selected.")
            # Clear the other field
            cleaned_data['price'] = None
            # Clear dimensions as they are hidden
            cleaned_data['height'] = None
            cleaned_data['width'] = None
        else:
            if not price and not price_per_m2:
                raise forms.ValidationError("You must provide either a Total Price or a Price per m².")
        
        return cleaned_data

class CategoryForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = ['name', 'icon']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'Category name'
            }),
            'icon': forms.FileInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none'
            }),
        }

class CompanyForm(forms.ModelForm):
    class Meta:
        model = Company
        fields = ['name', 'description', 'location', 'telegram_link', 'instagram_link', 'logo', 'is_active', 'subscription_deadline']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'Company Name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'Tell your story...',
                'rows': 4
            }),
            'location': forms.TextInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'e.g. Tashkent, Uzbekistan'
            }),
            'telegram_link': forms.TextInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': '@username or link'
            }),
            'instagram_link': forms.TextInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': '@username or link'
            }),
            'logo': forms.FileInput(attrs={
                'class': 'w-full text-xs text-outline file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-[10px] file:font-black file:bg-primary/10 file:text-primary hover:file:bg-primary/20'
            }),
            'subscription_deadline': forms.DateTimeInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'type': 'datetime-local'
            }),
        }

class HomeBannerForm(forms.ModelForm):
    class Meta:
        model = HomeBanner
        fields = ['title', 'subtitle', 'image', 'is_active', 'order']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'Main Title'
            }),
            'subtitle': forms.Textarea(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'Secondary Text',
                'rows': 3
            }),
            'image': forms.FileInput(attrs={
                'class': 'w-full text-xs text-outline file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-[10px] file:font-black file:bg-primary/10 file:text-primary hover:file:bg-primary/20'
            }),
            'order': forms.NumberInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
            }),
        }
from .models import Product, Category, Company, HomeBanner, LeadRequest

class LeadRequestForm(forms.ModelForm):
    class Meta:
        model = LeadRequest
        fields = ['lead_type', 'message', 'phone']
        widgets = {
            'lead_type': forms.HiddenInput(),
            'message': forms.Textarea(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': 'Special requests or preferences...',
                'rows': 3
            }),
            'phone': forms.TextInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl py-3 px-4 text-on-surface focus:ring-2 focus:ring-primary/10 outline-none',
                'placeholder': '+998 90 123 45 67'
            }),
        }
