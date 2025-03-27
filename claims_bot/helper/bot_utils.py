import json


def parse_message_response(response_text):
    try:
        json_pattern_start = response_text.find("```json")
        if json_pattern_start >= 0:
            json_start = response_text.find("{", json_pattern_start)
            backtick_end = response_text.find("```", json_start)
            json_end = response_text.rfind("}", json_start, backtick_end) + 1
        else:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            json_str = json_str.strip()
            response_data = json.loads(json_str)
        else:
            response_data = {
                "response": response_text,
                "response_type": "text",
                "information_extracted": {},
                "next_information_needed": None,
                "all_information_received": False,
            }
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        response_data = {
            "response": response_text,
            "response_type": "text",
            "information_extracted": {},
            "next_information_needed": None,
            "all_information_received": False,
        }

    return response_data


def parse_fraud_response(response_text):
    try:
        json_pattern_start = response_text.find("```json")
        if json_pattern_start >= 0:
            json_start = response_text.find("{", json_pattern_start)
            backtick_end = response_text.find("```", json_start)
            json_end = response_text.rfind("}", json_start, backtick_end) + 1
        else:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            json_str = json_str.strip()
            fraud_data = json.loads(json_str)

            if not all(key in fraud_data for key in ['fraud_score', 'fraud_flag', 'fraud_reasons']):
                print("Warning: Some expected fraud detection fields are missing")
        else:
            fraud_data = {
                "fraud_score": 0,
                "fraud_flag": False,
                "fraud_reasons": "No fraud indicators detected (default - failed to parse response)",
            }
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        fraud_data = {
            "fraud_score": 0,
            "fraud_flag": False,
            "fraud_reasons": "No fraud indicators detected (default - JSON parsing error)",
        }

    return fraud_data
