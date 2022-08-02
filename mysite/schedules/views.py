from rest_framework.generics import RetrieveUpdateDestroyAPIView, ListCreateAPIView
# from rest_framework.permissions import IsAuthenticated

from .serilizers import ScheduleSerializer
from .models import Schedule


class ListCreateScheduleAPIView(ListCreateAPIView):
    serializer_class = ScheduleSerializer
    queryset = Schedule.objects.all()
    # permission_classes = [IsAuthenticated]

    # def perform_create(self, serializer):
    #     # Assign the user who created the movie
    #     serializer.save(creator=self.request.user)


class RetrieveUpdateDestroyScheduleAPIView(RetrieveUpdateDestroyAPIView):
    serializer_class = ScheduleSerializer
    queryset = Schedule.objects.all()
