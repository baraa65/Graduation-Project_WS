from rest_framework.generics import RetrieveUpdateDestroyAPIView, ListCreateAPIView
# from rest_framework.permissions import IsAuthenticated

from .serilizers import ImageSerializer
from .models import Image


class ListCreateImageAPIView(ListCreateAPIView):
    serializer_class = ImageSerializer
    queryset = Image.objects.all()
    # permission_classes = [IsAuthenticated]

    # def perform_create(self, serializer):
    #     # Assign the user who created the movie
    #     serializer.save(creator=self.request.user)


class RetrieveUpdateDestroyImageAPIView(RetrieveUpdateDestroyAPIView):
    serializer_class = ImageSerializer
    queryset = Image.objects.all()
