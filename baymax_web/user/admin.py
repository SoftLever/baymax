from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User


class CustomUserAdmin(UserAdmin):
    readonly_fields = ('email', )


admin.site.register(User, CustomUserAdmin)
