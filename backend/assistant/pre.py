import requests
from bs4 import BeautifulSoup
import re

def clean_text(text):
    # \n과 \t를 공백으로 대체하고, 연속된 공백을 하나로 줄임
    cleaned_text = re.sub(r'\s+', ' ', text)
    # cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

    # 텍스트 양 끝의 공백 제거
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def crawl(query):
    # 요청할 URL
    url = f'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query={query}'

    # URL로부터 HTML 소스를 가져옴
    response = requests.get(url)

    # HTTP 요청이 성공했는지 확인
    if response.status_code == 200:
        # BeautifulSoup으로 HTML 로드
        soup = BeautifulSoup(response.content, 'lxml')
        
        # 'main_pack' ID를 가진 요소를 찾음
        main_pack = soup.find(id='main_pack')
        
        if main_pack:
            # 해당 요소 내의 모든 텍스트를 추출
            text_only = main_pack.get_text(separator='\n').strip()
            
            # 텍스트 전처리
            cleaned_text = clean_text(text_only)
            cleaned_text = cleaned_text
            with open('final_extracted_text.txt', 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
            
            # print("텍스트가 'final_extracted_text.txt' 파일에 저장되었습니다.")
            return cleaned_text
            # 텍스트 파일로 저장

        else:
            print("Element with id 'main_pack' not found.")
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        
if __name__ == '__main__':
    query = '일론머스크 재산'
    crawl(query)
    
