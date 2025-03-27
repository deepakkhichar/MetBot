CLAIMS_REQUIRED_INFO = {
    "incident_date": "date of the accident or incident",
    "incident_description": "detailed description of what happened",
    "vehicle_details": "make, model, year, and license plate of the vehicle",
    "damage_description": "description of the damages to the vehicle",
    "policy_number": "insurance policy number",
    "driver_info": "name and contact information of the driver",
    "location": "where the incident occurred",
    "other_parties": "information about other vehicles or parties involved, if any",
    "police_report": "whether a police report was filed and report number if available",
    "witnesses": "information about any witnesses",
}

CLAIMS_REQUIRED_DOCUMENTS = {
    "photos": "photos showing the damage to the vehicle",
    "police_report": "copy of the police report if available",
    "repair_estimate": "estimate from a repair shop if available",
    "driver_license": "copy of the driver's license",
}


SAMPLE_USERS = {
    'deepak': {
        'full_name': 'Deepak',
        'policy_number': 'POL-123456',
        'vehicle_make': 'Toyota',
        'vehicle_model': 'Camry',
        'vehicle_year': '2020',
        'license_plate': 'ABC123',
        'email': 'deepak@gmail.com',
        'phone_number': '555-123-4567',
    },
    'rajiv': {
        'full_name': 'Rajiv',
        'policy_number': 'POL-789012',
        'vehicle_make': 'Honda',
        'vehicle_model': 'Civic',
        'vehicle_year': '2019',
        'license_plate': 'XYZ789',
        'email': 'rajiv@gmail.com',
        'phone_number': '555-987-6543',
    },
    'piyush': {
        'full_name': 'Piyush',
        'policy_number': 'POL-345678',
        'vehicle_make': 'Ford',
        'vehicle_model': 'Mustang',
        'vehicle_year': '2021',
        'license_plate': 'MJK456',
        'email': 'piyush@gmail.com',
        'phone_number': '555-456-7890',
    },
}
CURRENT_USER = 'deepak'
