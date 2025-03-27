import json

import google.generativeai as genai
from django.conf import settings

from claims_bot.constants import CLAIMS_REQUIRED_DOCUMENTS, CLAIMS_REQUIRED_INFO
from claims_bot.helper.bot_utils import parse_message_response
from claims_bot.models import ClaimInformation, Message
from claims_bot.prompt import fraud_prompt, GEMINI_PROMPT_GUIDELINES


class GeminiService:
    def __init__(self):
        genai.configure(api_key=settings.ENV("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def process_message(self, conversation, user_message):
        """Process the user message and return a bot response."""
        messages = Message.objects.filter(conversation=conversation).order_by('created_at')

        claim_info, created = ClaimInformation.objects.get_or_create(conversation=conversation)

        formatted_history = self._format_chat_history(messages, claim_info)

        try:
            response = self.model.generate_content(formatted_history)
            response_text = response.text

            response_data = parse_message_response(response_text)

            if response_data.get("information_extracted"):
                for key, value in response_data["information_extracted"].items():
                    if hasattr(claim_info, key):
                        setattr(claim_info, key, value)
                claim_info.save()

            bot_response = {
                "content": response_data["response"],
                "response_type": response_data["response_type"],
                "information_key": response_data.get("next_information_needed"),
                "all_information_received": response_data.get("all_information_received", False),
            }

            return bot_response

        except Exception as e:
            return {
                "content": f"I'm sorry, I encountered an error: {str(e)}. Please try again.",
                "response_type": "text",
                "information_key": None,
                "all_information_received": False,
            }

    def fraud_detection(self, claim_info):

        fraud_detection_prompt = fraud_prompt(claim_info)
        fraud_response = self.model.generate_content(fraud_detection_prompt)
        fraud_result = parse_message_response(fraud_response.text)
        return fraud_result

    def _format_chat_history(self, messages, claim_info):
        """Format the chat history for Gemini API."""
        formatted_messages = []

        claim_info_dict = {
            field.name: getattr(claim_info, field.name)
            for field in claim_info._meta.fields
            if field.name != 'id' and field.name != 'conversation'
        }

        claim_info_dict = {k: v for k, v in claim_info_dict.items() if v is not None}

        missing_info = [
            key for key in CLAIMS_REQUIRED_INFO.keys() if key not in claim_info_dict or not claim_info_dict[key]
        ]

        from claims_bot.models import Document

        documents_uploaded = Document.objects.filter(message__conversation=claim_info.conversation).exists()

        if not documents_uploaded:
            missing_info.append("required_documents")

        system_prompt = f"""
        You are an AI assistant for a car insurance company helping customers file claims.

        Required information for a claim:
        {json.dumps(CLAIMS_REQUIRED_INFO, indent=2)}

        Required documents for a claim:
        {json.dumps(CLAIMS_REQUIRED_DOCUMENTS, indent=2)}

        Information collected so far:
        {json.dumps(claim_info_dict, indent=2)}

        Missing information:
        {json.dumps(missing_info, indent=2)}

        Documents uploaded: {"Yes" if documents_uploaded else "No"}

        IMPORTANT: Do not mark the claim as complete until all required information AND at least one document has been uploaded.

        {GEMINI_PROMPT_GUIDELINES}
        """

        formatted_messages.append({"role": "model", "parts": [{"text": system_prompt}]})

        for message in messages:
            role = "user" if message.role == "user" else "model"
            content = message.content

            formatted_messages.append({"role": role, "parts": [{"text": content}]})

        return formatted_messages
