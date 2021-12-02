from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('home/', views.home, name='home'),
    path('predict/upload', views.upload, name='upload'),  
    path('predict/', views.load_table, name='load'),
    path('predict/delselection', views.del_images, name='del_images'),
    path('predict/predict', views.predict, name='predict'),
    path('user-guide/', views.view_guide, name='view_guide'),
    path('predict/export', views.export, name='export')
]
