import gradio as gr
import pandas as pd
import yfinance as yf  # 주가 데이터를 가져오기 위한 라이브러리
import requests
from transformers import pipeline
import datetime
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 1. Sentiment Analysis 모델 로드
classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True, max_length=512)

# 2. OpenAI API 설정
OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

# 3. NewsAPI 설정
NEWS_API_KEY = ""

# 4. 이메일 설정
SENDER_EMAIL = ''
SENDER_PASSWORD = ''
RECEIVER_EMAIL = 'nick921@naver.com'

# 5. 뉴스 가져오기 함수 (개선된 버전)
def get_latest_stock_news(company_name, num_articles=10):
    try:
        # 주요 검색 키워드 설정
        keywords = [company_name, "금리", "환율", "경제 지표", "연준", "FED", "CPI", "통화 정책", "유가", "GDP"]
        keyword_query = " OR ".join(keywords)

        # API 요청: 제목에 특정 키워드 포함
        url = f"https://newsapi.org/v2/everything?qInTitle={keyword_query}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        data = response.json()

        # 요청 실패 또는 결과 없음
        if response.status_code != 200 or "articles" not in data or len(data["articles"]) == 0:
            return "뉴스 기사를 가져오는 데 실패했습니다.", []

        # 기사 필터링
        articles = []
        for i, article in enumerate(data["articles"][:num_articles]):
            title = article["title"]
            content = article["description"] or article["content"] or "내용 없음"
            link = article["url"]

            # 키워드 기반 필터링
            if any(keyword.lower() in (title + content).lower() for keyword in keywords):
                articles.append({"title": title, "content": content, "link": link})

        if not articles:
            return f"관련된 기사를 찾을 수 없습니다.", []

        return articles
    except Exception as e:
        return f"뉴스 API 오류: {e}", []


# 6. 주가 데이터 가져오기 함수
def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("주가 데이터를 찾을 수 없습니다.")
        return data
    except Exception as e:
        return None

# 7. GPT를 통한 주가 데이터 요약 함수
def gpt_summarize_stock_data(stock_data, ticker, period):
    if stock_data is None or stock_data.empty:
        return "주가 데이터를 불러오는 데 실패했습니다."

    stock_text = stock_data.tail(10).to_string()  # 최근 10일 데이터를 문자열로 변환
    prompt = f"""
    다음은 {ticker} 주식의 최근 {period}일 주가 데이터입니다. 이 데이터를 바탕으로 주가 동향을 요약해 주세요:

    {stock_text}

    요약:
    """
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        summary = ""
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                summary += chunk.choices[0].delta.content
        return summary.strip()
    except Exception as e:
        return f"GPT API 오류: {e}"

# 8. GPT 기반 투자 전략 생성 함수
def gpt_investment_strategy(news_text, stock_summary, ticker):
    try:
        prompt = f"""
        다음은 {ticker} 주식에 대한 최근 뉴스 및 주가 데이터 요약입니다.
        이를 기반으로 미래에 대한 AI투자 전략을 작성해 주세요.

        [주가 데이터 요약]
        {stock_summary}

        [뉴스 정보]
        {news_text}

        투자 전략:
        """
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        strategy = ""
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                strategy += chunk.choices[0].delta.content
        return strategy.strip()
    except Exception as e:
        return f"GPT API 오류: {e}"
# 9. 자동 투자 분석 함수
def investment_analysis(company_name, ticker, period_days):
    try:
        # 입력 값 검증
        period_days = int(period_days)  # 텍스트로 입력된 기간을 정수로 변환
        if period_days <= 0:
            raise ValueError("기간은 양의 정수여야 합니다.")

        # 뉴스 가져오기
        articles = get_latest_stock_news(company_name, num_articles=5)
        if isinstance(articles, str):
            raise ValueError("뉴스 데이터가 없습니다.")

        # 주가 데이터 가져오기
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=period_days)
        stock_data = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if stock_data is None:
            raise ValueError("주가 데이터를 가져오는 데 실패했습니다.")

        # 주가 데이터 요약 (GPT 사용)
        stock_summary = gpt_summarize_stock_data(stock_data, ticker, period_days)

        # 뉴스 기사 데이터 표시
        news_text = "\n\n".join([f"[{i + 1}] 제목: {article['title']}\n내용: {article['content']}\n링크: {article['link']}" for i, article in enumerate(articles)])

        # GPT 투자 전략 생성
        strategy = gpt_investment_strategy(news_text, stock_summary, ticker)

        # 주가 데이터 시각화
        chart = stock_data[['Close']].plot(title=f"{ticker} 주가 추이 ({start_date} ~ {end_date})", figsize=(8, 4)).get_figure()

        # 결과 텍스트 구성
        news_result_text = f"[뉴스 정보]\n{news_text}\n\n[기사 수: {len(articles)}개]"
        stock_result_text = f"[주가 데이터 요약]\n{stock_summary}"

        return news_result_text, stock_result_text, chart, strategy
    except Exception as e:
        return f"오류 발생: {e}", f"오류 발생: {e}", None, "투자 전략을 생성할 수 없습니다."

# 10. GPT 기반 티커 검색 함수
def gpt_get_ticker(company_name):
    try:
        prompt = f"""
        "{company_name}"라는 회사 이름에 대해 적절한 주식 티커를 반환해 주세요.
        입력된 이름이 한글이라면, 이를 영어로 번역한 후 검색하고 티커를 반환해 주세요.
        관련 정보가 없을 경우, "정보 없음"이라고 답변해 주세요.
        반환 값은 반드시 한 단어로 출력해 주세요.
        """
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        ticker = ""
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                ticker += chunk.choices[0].delta.content
        return ticker.strip()
    except Exception as e:
        return f"GPT API 오류: {e}"

# 11. 이메일 전송 함수
def send_email(company_name, strategy):
    try:
        subject = f"{company_name} 투자 전략 보고서"
        body = f"""
        [회사 이름]: {company_name}

        [GPT 투자 전략]
        {strategy}
        """
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        return "이메일 전송 성공"
    except Exception as e:
        return f"이메일 전송 오류: {e}"

# 12. Gradio UI 생성
def create_interface():
    with gr.Blocks(css="""
        body { font-family: 'Roboto', sans-serif; background-color: #f7f9fc; color: #333; }
        .gr-textbox { font-size: 14px; border: 1px solid #ddd; padding: 12px; border-radius: 8px; background-color: #fff; }
        .blue-button {
            background-color: #5DADE2 !important;  /* 연한 파란색 */
            color: white !important;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
        }
        .blue-button:hover {
            background-color: #3498DB !important;  /* 호버 시 조금 더 짙은 파란색 */
        }
    """) as demo:
        gr.Markdown("# 🚀 **미국 주식 투자 분석 AI AGENT**\n"
                    "💡 **해당 주식 관련 뉴스 정보**, **주식 티커(종목 코드)**, **분석 원하는 차트 기간**을 바탕으로 투자 전략을 제공해 드립니다!\n\n"
                    "⬇️ 아래에서 분석을 시작하세요.")

        with gr.Tab("📊 투자 분석"):
            with gr.Row(equal_height=True):
                company_name_input = gr.Textbox(label="뉴스 검색 키워드 🏢 (예: Tesla, Apple)", lines=1, placeholder="예: Tesla")
                ticker_input = gr.Textbox(label="주식 티커(종목코드) 입력 💹 (예: TSLA, AAPL)", lines=1, placeholder="예: TSLA")
                period_input = gr.Textbox(label="차트 분석 기간 입력 📅 (일 단위, 예: 30)", value="30", lines=1, placeholder="30")

            # 버튼에 elem_classes를 사용하여 클래스를 추가
            analyze_button = gr.Button("🔍 분석 실행", elem_classes=["blue-button"])
            with gr.Row():
                news_output = gr.Textbox(label="📰 뉴스 정보", lines=10, interactive=False, placeholder="해당 키워드가 포함된 뉴스 기사 5개가 여기에 표시됩니다.")
                chart_output = gr.Plot(label="📈 주가 차트")
                stock_summary_output = gr.Textbox(label="📊 주가 데이터 차트 요약", lines=10, interactive=False, placeholder="주가 데이터 차트 요약이 여기에 표시됩니다.")

            strategy_output = gr.Textbox(label="GPT 기반 투자 전략 📑", lines=8, interactive=False, placeholder="GPT가 생성한 투자 전략이 여기에 표시됩니다.")
            email_button = gr.Button("📧 이메일로 투자 전략 전송 받기", elem_classes=["blue-button"])
            email_status_output = gr.Textbox(label="이메일 전송 상태 ✅", interactive=False, placeholder="이메일 전송 상태가 여기에 표시됩니다.")

            analyze_button.click(investment_analysis, inputs=[company_name_input, ticker_input, period_input],
                                 outputs=[news_output, stock_summary_output, chart_output, strategy_output])
            email_button.click(send_email, inputs=[company_name_input, strategy_output], outputs=[email_status_output])

        with gr.Tab("🔎 티커(종목 코드) 검색"):
            company_name_input_ticker = gr.Textbox(label="회사 이름 입력 🏢", placeholder="예: Microsoft")
            ticker_output = gr.Textbox(label="티커(종목 코드) 반환 💹", interactive=False, placeholder="회사의 티커가 여기에 표시됩니다.")
            search_button = gr.Button("🔍 티커 검색", elem_classes=["blue-button"])

            search_button.click(gpt_get_ticker, inputs=[company_name_input_ticker], outputs=[ticker_output])

        demo.launch()



# 13. 프로그램 실행
if __name__ == "__main__":
    create_interface()