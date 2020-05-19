from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('search/', views.index, name='search'),
    path('add/', views.add, name='add'),
    path('scheck/', views.sCheck, name='scheck')
    #path('showplt/', views.graph_data, name='showplt'),

]
