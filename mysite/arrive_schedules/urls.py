from django.urls import path
from . import views


urlpatterns = [
    path('', views.ListCreateScheduleAPIView.as_view(), name='get_post_schedules'),
    path('<int:pk>/', views.RetrieveUpdateDestroyScheduleAPIView.as_view(), name='get_delete_update_schedule'),
]