from django.contrib.auth import get_user_model
from user.serializers import UserSerializer, StaffUserSerializer
from user.permissions import IsAdminOrIsSelf
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework import status
from rest_framework.decorators import action


User = get_user_model()


class UserViewSet(viewsets.ModelViewSet):

    queryset = User.objects.all()

    def list(self, request):
        serializer = self.get_serializer(self.queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        user = self.get_object()
        serializer = self.get_serializer(user)
        return Response(serializer.data)

    def create(self, request):
        data = request.data
        if isinstance(data, list):
            serializer = self.get_serializer(data=data, many=True)
        else:
            serializer = self.get_serializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(status=status.HTTP_201_CREATED)
        return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None):
        data = request.data
        user = self.get_object()
        serializer = self.get_serializer(user, data=data)

        if serializer.is_valid():
            serializer.save()
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def partial_update(self, request, pk=None):
        data = request.data
        user = self.get_object()
        serializer = self.get_serializer(user, data=data, partial=True)

        if serializer.is_valid():
            serializer.save()
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        user = self.get_object()
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=True, methods=['patch'], permission_classes=[IsAdminOrIsSelf], url_path='set-password')
    def set_password(self, request, pk=None):
        user = self.get_object()
        password = request.data.get('password')
        if password:
            user.set_password(password)
            user.save()
            return Response({'status': 'Password Set'}, status=status.HTTP_200_OK)

        return Response(status=status.HTTP_400_BAD_REQUEST)

    def get_permissions(self):
        if self.action in ['list', 'create', 'destroy']:
            permission_classes = [IsAdminUser, ]
        else:
            permission_classes = [IsAuthenticated, IsAdminOrIsSelf]
        return [permission() for permission in permission_classes]

    def get_serializer(self, *args, **kwargs):
        if self.request.user.is_staff and self.request.user.is_active:
            return StaffUserSerializer(*args, **kwargs)
        return UserSerializer(*args, **kwargs)
