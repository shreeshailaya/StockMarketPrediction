from django.db import models
from django import forms


class PortfolioModels(models.Model):
    stock_name = models.CharField(max_length=100)
    stock_price = models.CharField(max_length=100)

    def __str__(self):
        return self.stock_name
