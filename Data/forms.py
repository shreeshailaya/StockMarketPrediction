from django import forms
from .models import PortfolioModels


class Portfolio(forms.ModelForm):
    class Meta:
        model = PortfolioModels
        fields = ["stock_name", "stock_price"]
