import hmac
import hashlib
import json
from urllib.parse import parse_qsl

def verify_telegram_webapp_data(init_data: str, bot_token: str) -> dict:
    """
    Verifies the data received from the Telegram WebApp.
    Returns the parsed user data if valid, otherwise None.
    """
    try:
        # Parse query string
        parsed_data = dict(parse_qsl(init_data))
        if 'hash' not in parsed_data:
            return None
        
        received_hash = parsed_data.pop('hash')
        
        # Sort keys and construct data_check_string
        data_check_string = '\n'.join([f"{k}={v}" for k, v in sorted(parsed_data.items())])
        
        # HMAC-SHA256 signature verification
        secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
        computed_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
        
        if computed_hash != received_hash:
            return None
            
        # Parse the 'user' field which is a JSON string
        if 'user' in parsed_data:
            return json.loads(parsed_data['user'])
        
        return parsed_data
    except Exception as e:
        print(f"Auth error: {e}")
        return None
