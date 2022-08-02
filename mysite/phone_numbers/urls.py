from django.urls import path
from . import views


urlpatterns = [
    path('', views.ListCreateNumberAPIView.as_view(), name='get_post_numbers'),
    path('<int:pk>/', views.RetrieveUpdateDestroyNumberAPIView.as_view(), name='get_delete_update_number'),
]