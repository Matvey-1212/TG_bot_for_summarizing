import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json
import requests
import time

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

params_config = {    
                "table": os.getenv("DB_TABLE"),
                "limit": os.getenv("DB_limit"),
                }

API_URL = f"http://localhost:{os.getenv('APP_PORT')}/predict"

def get_db_connection(conf):
    return psycopg2.connect(**conf)

def fetch_unprocessed_news(conn, table, limit):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, message FROM public.{table}
            WHERE summary IS NULL AND message IS NOT NULL
            ORDER BY published_at DESC
            LIMIT {limit};
        """)
        return cur.fetchall()


def get_summary_from_model(text):
    try:
        response = requests.post(API_URL, json={"text": text})
        response.raise_for_status()
        return response.json().get("summary")
    except Exception as e:
        print(f"Ошибка при вызове модели: {e}")
        return None
    
def get_summary_from_model(texts, api_url=None):
    """
    Получает суммаризацию текстов от модели через API
    """
    
    try:
        input_data = [{"news_id": nid, "text": text} for nid, text in texts]
        
        response = requests.post(
            api_url,
            json=input_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к модели: {e}")
        return None
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return None

def update_summary(conn, table, news_id, summary, params): #
    with conn.cursor() as cur:
        json_data = [{
            'ru_content': 0,
            'ru_title': params
        }]
        cur.execute(f"""
            UPDATE public.{table}
            SET summary = '{summary}', params = {Json(json_data)}
            WHERE id = {news_id};
        """)
    conn.commit()

def main():
    while True:
        try:
            conn = get_db_connection(DB_CONFIG)
            news_batch = fetch_unprocessed_news(conn, table=params_config['table'], limit=params_config['limit'])

            if news_batch:
                print(f"Обрабатывается {len(news_batch)} новостей...")
                result_sum = get_summary_from_model(news_batch, api_url=API_URL)
                for res in result_sum:
                    news_id = res['news_id']
                    summary = res['summary']
                    params = res['sum_class']
                    update_summary(conn, 
                                    table=params_config['table'], 
                                    news_id = news_id, 
                                    summary = summary,
                                    params = params)
                print('Конец обработки')


            conn.close()
        except Exception as e:
            print(f"Ошибка: {e}")
        
        time.sleep(3)

if __name__ == "__main__":
    main()
