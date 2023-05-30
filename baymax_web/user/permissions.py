from rest_framework import permissions


class IsAdminOrIsSelf(permissions.BasePermission):
    """
    Custom permission to only allow owners of an object to edit it.
    """

    def has_object_permission(self, request, view, obj):
        return obj == request.user or request.user.is_staff
