import os
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

from supabase import create_client, Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    # 2. Create document in Supabase 'alerts' table
    alert_data = {
        "type": "Test Alert",
        "priority": "LOW",
        "image_url": "https://example.com/dummy.jpg",
        "confidence": 0.99,
        "instruction": "This is a test alert."
    }
    supabase.table("alerts").insert(alert_data).execute()
    print("✅ Inserted successfully into table 'alerts'!")
except Exception as e:
    print(f"❌ Insert failed: {e}")
