from rest_framework.generics import RetrieveUpdateDestroyAPIView, ListCreateAPIView
# from rest_framework.permissions import IsAuthenticated

from .serilizers import NumberSerializer
from .models import Number


class ListCreateNumberAPIView(ListCreateAPIView):
    serializer_class = NumberSerializer
    queryset = Number.objects.all()
    # permission_classes = [IsAuthenticated]

    # def perform_create(self, serializer):
    #     # Assign the user who created the movie
    #     serializer.save(creator=self.request.user)


class RetrieveUpdateDestroyNumberAPIView(RetrieveUpdateDestroyAPIView):
    serializer_class = NumberSerializer
    queryset = Number.objects.all()
