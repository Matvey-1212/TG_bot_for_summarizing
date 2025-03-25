import requests
import re
import csv
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import datetime

def get_archive_url(date):
    """根据传入的日期构造归档页面的 URL，格式为 https://lenta.ru/YYYY/MM/DD/"""
    date_str = date.strftime("%Y/%m/%d/")
    archive_url = f"https://lenta.ru/{date_str}"
    return archive_url

def get_date_str(date):
    """格式化日期为 YYYY-MM-DD"""
    return date.strftime("%Y-%m-%d")

def crawl_archive_page(archive_url):
    """爬取给定归档页面中的所有新闻条目"""
    base_url = "https://lenta.ru"
    try:
        response = requests.get(archive_url)
        response.raise_for_status()
    except requests.RequestException as e:
        print("获取页面失败:", e)
        return []

    soup = BeautifulSoup(response.content, 'lxml')
    articles = soup.find_all("a", class_="card-full-news _archive")
    results = []

    for article in articles:
        relative_link = article.get("href")
        absolute_link = urljoin(base_url, relative_link)
        title_tag = article.find("h3", class_="card-full-news__title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        time_tag = article.find("time", class_="card-full-news__date")
        time_text = time_tag.get_text(strip=True) if time_tag else ""
        rubric_tag = article.find("span", class_="card-full-news__rubric")
        rubric_text = rubric_tag.get_text(strip=True) if rubric_tag else ""

        results.append({
            "title": title,
            "time": time_text,  # 例如 "00:16"
            "rubric": rubric_text,
            "url": absolute_link
        })
    return results

def extract_description(detail_url):
    """从新闻详情页中提取 description 字段"""
    try:
        response = requests.get(detail_url)
        response.raise_for_status()
    except requests.RequestException as e:
        print("获取详情页失败:", e)
        return None

    # 使用正则表达式查找 "description" 字段
    pattern = r'"description":\s*"([^"]+)"'
    match = re.search(pattern, response.text)
    if match:
        return match.group(1)
    else:
        print("未在页面中匹配到 description 字段：", detail_url)
        return None

def save_to_csv(news_list, filename):
    """将新闻数据保存到 CSV 文件中"""
    headers = ["Заголовок", "Текст", "Время", "Раздел"]

    with open(filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for news in news_list:
            # 将日期 (YYYY-MM-DD) 与当天新闻中的时间拼接
            full_time = f"{news['date']} {news['time']}"
            writer.writerow([news["title"], news["description"], full_time, news["rubric"]])

if __name__ == "__main__":
    # 当前日期
    today = datetime.date.today()
    # 定义两个月前的日期（这里大约按 60 天计算）
    start_date = today - datetime.timedelta(days=60)

    current_date = start_date
    all_news = []
    while current_date <= today:
        print("处理日期:", current_date)
        archive_url = get_archive_url(current_date)
        articles = crawl_archive_page(archive_url)
        date_str = get_date_str(current_date)
        for article in articles:
            print("处理新闻:", article["title"])
            description = extract_description(article["url"])
            if description:
                news_item = {
                    "title": article["title"],
                    "description": description,
                    "time": article["time"],
                    "rubric": article["rubric"],
                    "date": date_str
                }
                all_news.append(news_item)
            else:
                print("未能提取正文内容：", article["url"])
        # 处理下一天
        current_date += datetime.timedelta(days=1)

    if all_news:
        output_filename = "lenta_news.csv"
        save_to_csv(all_news, output_filename)
        print("成功保存到 CSV 文件：", output_filename)
    else:
        print("未提取到任何新闻数据。")
