from django.contrib import admin

from claims_bot.models import ClaimInformation, Conversation, Document, Message


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('user', 'created_at', 'updated_at', 'completed')


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('conversation', 'role', 'content')


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('message',)




@admin.register(ClaimInformation)
class ClaimInformationAdmin(admin.ModelAdmin):
    list_display = ('conversation',)
