from django.urls import path
from . import views


urlpatterns = [
    path('', views.ListCreateImageAPIView.as_view(), name='get_post_images'),
    path('<int:pk>/', views.RetrieveUpdateDestroyImageAPIView.as_view(), name='get_delete_update_image'),
]