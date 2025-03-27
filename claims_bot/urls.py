from django.urls import include, path
from rest_framework.routers import DefaultRouter

from claims_bot.views import chat_home, claim_summary, ConversationViewSet, transcribe_audio

router = DefaultRouter()
router.register(r'conversations', ConversationViewSet)

urlpatterns = [
    path('', chat_home, name='chat_home'),
    path('claim-summary/<int:conversation_id>/', claim_summary, name='claim_summary'),
    path('api/', include(router.urls)),
    path('api/transcribe/', transcribe_audio, name='transcribe_audio'),
]
