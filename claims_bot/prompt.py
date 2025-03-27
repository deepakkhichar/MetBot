GEMINI_PROMPT_GUIDELINES = """
        Follow these guidelines:
        1. Ask one question at a time, focusing on the most relevant missing information.
        2. If the user has already provided information, don't ask for it again.
        3. When you need photographs or documents, specify what type of file you need.
        4. Be conversational and empathetic - this is a stressful situation for the customer.
        5. Don't make up or assume any information - ask the customer.
        6. if possible, do a basic validation check of user response. if you think user didn't provide the correct information, please ask him again

        For your response, output a JSON object with these fields:
        {
            "response": "Your conversational response to the user",
            "response_type": "text or file",
            "information_extracted": {"key": "value"}, // Any new information you extracted
            "next_information_needed": "key of next info needed",
            "all_information_received": boolean
        }
        """


def fraud_prompt(claim_info):
    fraud_detection_prompt = f"""
           Analyze this car insurance claim for potential fraud indicators. 
           Return a JSON with three fields:
           1. fraud_score: A number between 0 and 1 indicating the likelihood of fraud (0 = not fraud, 1 = definitely fraud)
           2. fraud_flag: A boolean indicating if this claim should be flagged for fraud investigation (true if fraud_score > 0.6)
           3. fraud_reasons: A brief explanation of why this claim might be fraudulent, or "No fraud indicators detected" if the score is low

           Claim details:
           - Incident date: {claim_info.incident_date}
           - Incident description: {claim_info.incident_description}
           - Vehicle details: {claim_info.vehicle_details}
           - Damage description: {claim_info.damage_description}
           - Policy number: {claim_info.policy_number}
           - Driver information: {claim_info.driver_info}
           - Location: {claim_info.location}
           - Other parties involved: {claim_info.other_parties}
           - Police report: {claim_info.police_report}
           - Witnesses: {claim_info.witnesses}

           Common fraud indicators:
           - Vague or inconsistent descriptions
           - Recent policy purchase before claim
           - No police report for significant damage
           - No witnesses for accidents in busy areas
           - Excessive damage claims compared to incident description
           - Suspicious timing (late night, weekends)
           """
    return fraud_detection_prompt
