from django.urls import path
from .views import generate_recipe, store_preferences, get_chat_history, detect_ingredients

urlpatterns = [
    path('generate_recipe/', generate_recipe, name='generate_recipe'),
    path('store_preferences/', store_preferences, name='store_preferences'),
    path('get_chat_history/', get_chat_history, name='get_chat_history'),
    path('detect_ingredients/', detect_ingredients, name='detect_ingredients'),
]
