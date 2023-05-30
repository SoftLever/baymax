from rest_framework import serializers
from django.contrib.auth import get_user_model

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    id = serializers.UUIDField(
        read_only=True,
        required=False,
    )

    password = serializers.CharField(
        write_only=True,
        max_length=128,
        style={'input_type': 'password'},
        required=False,
    )

    last_login = serializers.DateTimeField(
        read_only=True,
    )

    date_joined = serializers.DateTimeField(
        read_only=True,
    )

    class Meta:
        model = User
        fields = ('id', 'username', 'first_name', 'last_name', 'password', 'last_login', 'date_joined')

    def create(self, validated_data):
        password = validated_data.pop('password', None)
        user = User(**validated_data)
        if password:
            user.set_password(password)
        user.save()
        return user

    def update(self, instance, validated_data):
        instance.first_name = validated_data.get('first_name', instance.first_name)
        instance.last_name = validated_data.get('last_name', instance.last_name)
        instance.username = validated_data.get('username', instance.username)
        instance.save()
        return instance


class StaffUserSerializer(UserSerializer):

    class Meta:
        model = User
        fields = '__all__'

    def update(self, instance, validated_data):
        instance.is_superuser = validated_data.get('is_superuser', instance.is_superuser)
        instance.first_name = validated_data.get('first_name', instance.first_name)
        instance.last_name = validated_data.get('last_name', instance.last_name)
        instance.is_staff = validated_data.get('is_staff', instance.is_staff)
        instance.is_active = validated_data.get('is_active', instance.is_active)
        instance.username = validated_data.get('username', instance.username)
        # instance.groups = validated_data.get('groups', instance.groups)
        # instance.user_permissions = validated_data.get('user_permissions', instance.user_permissions)
        instance.save()
        return instance
