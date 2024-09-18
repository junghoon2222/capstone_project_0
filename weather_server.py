from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
from bs4 import BeautifulSoup

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/get_weather")
async def get_result():
    response = requests.get('https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query=%EA%B5%AC%EB%AF%B8+%EB%82%A0%EC%94%A8')
    
    if response.status_code == 200:
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # 'main_pack' ID를 가진 요소를 찾음
        status_wrap_text = soup.find(class_='status_wrap').get_text()
        
        if main_pack:
            # 해당 요소 내의 모든 텍스트를 추출
            text_only = main_pack.get_text(separator='\n').strip()
            
            # 텍스트 전처리
            cleaned_text = clean_text(text_only)
            with open('final_extracted_text.txt', 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
            
            print("텍스트가 'final_extracted_text.txt' 파일에 저장되었습니다.")
            return cleaned_text
            # 텍스트 파일로 저장

        else:
            print("Element with id 'main_pack' not found.")
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        
    
    if result_queue:
        result = result_queue.popleft()
        return JSONResponse(content=result)
    return JSONResponse(content={})

if __name__ == '__main__':

    uvicorn.run(app, host='0.0.0.0', port=7002)