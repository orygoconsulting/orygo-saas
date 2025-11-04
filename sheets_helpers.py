import os
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path

SCOPE = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

def get_gsheet_client(service_account_json_path: str):
    if not Path(service_account_json_path).exists():
        raise FileNotFoundError("Google service account JSON no encontrado en: " + service_account_json_path)
    creds = ServiceAccountCredentials.from_json_keyfile_name(service_account_json_path, SCOPE)
    client = gspread.authorize(creds)
    return client

def read_sheet_as_df(client, spreadsheet_id: str, sheet_name: str):
    sh = client.open_by_key(spreadsheet_id)
    ws = sh.worksheet(sheet_name)
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    return df
