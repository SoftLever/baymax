from django.contrib.auth.signals import user_logged_in, user_logged_out
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import permissions
from django.contrib.auth import authenticate

from knox.auth import TokenAuthentication
from knox.models import AuthToken
from knox.settings import knox_settings


class LoginView(APIView):
    permission_classes = (permissions.AllowAny,)

    def post(self, request, format=None):
        user = authenticate(username=request.data.get('username'), password=request.data.get('password'))
        if not user:
            return Response(data={'status': "Invalid Credentials or User doesn't exist"}, status=status.HTTP_401_UNAUTHORIZED)
        token = AuthToken.objects.create(user)[1]  # https://stackoverflow.com/a/55881400/9366954
        user_logged_in.send(sender=user.__class__, request=request, user=user)
        serializer = knox_settings.USER_SERIALIZER
        context = {'request': self.request, 'format': self.format_kwarg, 'view': self}
        if serializer is None:
            return Response(
                {'token': token}
            )
        return Response({
            'user': serializer(user, context=context).data,
            'token': token,
        })


class LogoutView(APIView):
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)

    def post(self, request, format=None):
        request._auth.delete()
        user_logged_out.send(sender=request.user.__class__, request=request, user=request.user)
        return Response(None, status=status.HTTP_204_NO_CONTENT)


class LogoutAllView(APIView):
    """
    Log the user out of all sessions.
    I.E. deletes all auth tokens for the user
    """
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)

    def post(self, request, format=None):
        request.user.auth_token_set.all().delete()
        user_logged_out.send(sender=request.user.__class__, request=request, user=request.user)
        return Response(None, status=status.HTTP_204_NO_CONTENT)
