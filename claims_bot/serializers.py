from rest_framework import serializers
from claims_bot.models import Conversation, Message, Document, ClaimInformation


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'file', 'uploaded_at']
        read_only_fields = ['id', 'uploaded_at']


class MessageSerializer(serializers.ModelSerializer):
    documents = DocumentSerializer(many=True, read_only=True)
    
    class Meta:
        model = Message
        fields = ['id', 'role', 'content', 'response_type', 'information_key', 'created_at', 'documents']
        read_only_fields = ['id', 'created_at']


class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = Conversation
        fields = ['id', 'user', 'created_at', 'updated_at', 'completed', 'messages']
        read_only_fields = ['id', 'created_at', 'updated_at']


class ClaimInformationSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClaimInformation
        exclude = ['id', 'conversation']