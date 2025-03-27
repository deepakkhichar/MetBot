from django.db import models


class Conversation(models.Model):
    user = models.CharField(max_length=50, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed = models.BooleanField(default=False)

    def __str__(self):
        return f"Conversation {self.id}"


class Message(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('bot', 'Bot'),
    ]
    RESPONSE_TYPE_CHOICES = [
        ('text', 'Text'),
        ('file', 'File'),
    ]
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    response_type = models.CharField(
        max_length=10,
        default='text',
        choices=RESPONSE_TYPE_CHOICES,
    )
    information_key = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."


class Document(models.Model):
    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name='documents')
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Document for {self.message}"


class ClaimInformation(models.Model):
    conversation = models.OneToOneField(Conversation, on_delete=models.CASCADE, related_name='claim_information')
    incident_date = models.CharField(max_length=100, null=True, blank=True)
    incident_description = models.TextField(null=True, blank=True)
    vehicle_details = models.TextField(null=True, blank=True)
    damage_description = models.TextField(null=True, blank=True)
    policy_number = models.CharField(max_length=100, null=True, blank=True)
    driver_info = models.TextField(null=True, blank=True)
    location = models.CharField(max_length=255, null=True, blank=True)
    other_parties = models.TextField(null=True, blank=True)
    police_report = models.TextField(null=True, blank=True)
    witnesses = models.TextField(null=True, blank=True)

    fraud_score = models.FloatField(default=0.0)
    fraud_flag = models.BooleanField(default=False)
    fraud_reasons = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Claim Information for {self.conversation.id}"
