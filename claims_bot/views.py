import base64
import json
import os
import tempfile

import speech_recognition as sr
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from claims_bot.bot import GeminiService
from claims_bot.constants import CURRENT_USER, SAMPLE_USERS
from claims_bot.models import ClaimInformation, Conversation, Document, Message
from claims_bot.serializers import ClaimInformationSerializer, ConversationSerializer, MessageSerializer


def chat_home(request):
    """Render the chat interface"""
    return render(request, 'chat.html')


def claim_summary(request, conversation_id):
    """Render the claim summary page"""
    conversation = get_object_or_404(Conversation, id=conversation_id)
    claim_info = get_object_or_404(ClaimInformation, conversation=conversation)

    messages = Message.objects.filter(conversation=conversation)
    documents = []
    for message in messages:
        docs = Document.objects.filter(message=message)
        if docs:

            documents.extend(docs)
    return render(
        request,
        'claim_summary.html',
        {
            'conversation': conversation,
            'claim_info': claim_info,
            'documents': documents,
            'fraud_detected': claim_info.fraud_flag,
            'fraud_score': claim_info.fraud_score,
            'fraud_reasons': claim_info.fraud_reasons,
        },
    )


class ConversationViewSet(viewsets.ModelViewSet):
    queryset = Conversation.objects.all()
    serializer_class = ConversationSerializer

    def create(self, request):
        """Start a new conversation with just user information"""
        username = CURRENT_USER #request.data.get('user', None) #using default current user as of now
        conversation = Conversation.objects.create(user=username)

        initial_message = Message.objects.create(
            conversation=conversation,
            role='bot',
            content="Hello! I'm here to help you with your car insurance claim. Could you please describe what happened?",
            response_type='text',
        )

        claim_info = ClaimInformation.objects.create(conversation=conversation)

        if username in SAMPLE_USERS.keys():
            user_data = SAMPLE_USERS[username]
            claim_info.policy_number = user_data['policy_number']
            claim_info.vehicle_details = f"Make: {user_data['vehicle_make']}, Model: {user_data['vehicle_model']}, Year: {user_data['vehicle_year']}, License Plate: {user_data['license_plate']}"
            claim_info.driver_info = (
                f"Name: {user_data['full_name']}, Email: {user_data['email']}, Phone: {user_data['phone_number']}"
            )
            claim_info.save()
        return Response(
            {
                'conversation_id': conversation.id,
                'message': {
                    'id': initial_message.id,
                    'content': initial_message.content,
                    'role': initial_message.role,
                    'created_at': initial_message.created_at.isoformat(),
                },
            },
            status=status.HTTP_201_CREATED,
        )

    @action(detail=True, methods=['post'])
    def send_message(self, request, pk=None):
        """Process a new user message and generate bot response - handles both text and file uploads"""
        conversation = self.get_object()

        if conversation.completed:
            return Response({"error": "This conversation is already completed."}, status=status.HTTP_400_BAD_REQUEST)

        files = request.FILES.getlist('files')

        user_message = Message.objects.create(
            conversation=conversation, role='user', content=request.data.get('content', ''), response_type='text'
        )

        if files:
            for file in files:
                document_type = request.data.get('document_type', 'other')
                Document.objects.create(message=user_message, file=file)

                user_message.content = f"{user_message.content} (Uploaded {document_type})"
                user_message.save()

        gemini_service = GeminiService()
        bot_response = gemini_service.process_message(conversation, user_message)

        bot_message = Message.objects.create(
            conversation=conversation,
            role='bot',
            content=bot_response["content"],
            response_type=bot_response["response_type"],
            information_key=bot_response["information_key"],
        )

        if bot_response["all_information_received"]:
            conversation.completed = True
            conversation.save()

            claim_info = conversation.claim_information
            gemini_service = GeminiService()
            fraud_result = gemini_service.fraud_detection(claim_info)

            claim_info.fraud_score = fraud_result.get('fraud_score', 0.0)
            claim_info.fraud_flag = fraud_result.get('fraud_flag', False)
            claim_info.fraud_reasons = fraud_result.get('fraud_reasons', 'No fraud analysis available')
            claim_info.save()

            return Response(
                {
                    "all_information_received": True,
                    "redirect_url": (f'/claims_bot/claim-summary' f'/{conversation.id}/'),
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                'conversation_id': conversation.id,
                'user_message': MessageSerializer(user_message).data,
                'bot_message': MessageSerializer(bot_message).data,
                'all_information_received': bot_response["all_information_received"],
            }
        )

    @action(detail=True, methods=['get'])
    def claim_information(self, request, pk=None):
        """Get current claim information"""
        conversation = self.get_object()
        claim_info = get_object_or_404(ClaimInformation, conversation=conversation)

        serializer = ClaimInformationSerializer(claim_info)
        return Response(serializer.data)


@csrf_exempt
@require_POST
def transcribe_audio(request):
    """Transcribe audio to text using speech recognition"""
    try:
        from pydub import AudioSegment

        data = json.loads(request.body)
        audio_data = data.get('audio_data')

        if not audio_data:
            return JsonResponse({'error': 'No audio data provided'}, status=400)

        audio_bytes = base64.b64decode(audio_data)

        temp_input = tempfile.NamedTemporaryFile(delete=False)
        temp_input.write(audio_bytes)
        temp_input.close()

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_output.close()

        try:
            audio = AudioSegment.from_file(temp_input.name, format="mp3")
            audio.export(temp_output.name, format="wav")
        except Exception as conversion_error:
            try:
                audio = AudioSegment.from_file(temp_input.name)
                audio.export(temp_output.name, format="wav")
            except Exception as e:
                raise Exception(f"Could not convert audio: {str(e)}")

        recognizer = sr.Recognizer()

        with sr.AudioFile(temp_output.name) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)

        os.unlink(temp_input.name)
        os.unlink(temp_output.name)

        return JsonResponse({'text': text})

    except sr.UnknownValueError:
        return JsonResponse({'text': '', 'error': 'Could not understand audio'}, status=400)
    except sr.RequestError as e:
        return JsonResponse({'text': '', 'error': f'Error with the speech recognition service: {e}'}, status=500)
    except Exception as e:
        return JsonResponse({'text': '', 'error': str(e)}, status=500)
