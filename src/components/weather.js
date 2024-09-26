import React, { useState, useEffect } from "react";
import "./weather.css"; // 같은 경로의 CSS 파일 불러오기

function Weather() {
  const [weather, setWeather] = useState(null); // 날씨 데이터를 저장할 상태
  const [date, setDate] = useState(""); // 날짜 데이터를 저장할 상태

  // 날씨 데이터를 fetch하는 함수
  const fetchWeather = async () => {
    try {
      const weatherResponse = await fetch(
        `http://182.218.49.58:50006/get_weather`
      );
      if (!weatherResponse.ok) {
        throw new Error(`Error: ${weatherResponse.status}`);
      }
      const weatherData = await weatherResponse.json();
      setWeather(weatherData);
      console.log(weatherData);
    } catch (error) {
      console.error("Error fetching weather data", error);
      setWeather(null);
    }
  };

  // 컴포넌트가 처음 마운트되었을 때와 5분마다 날씨 데이터를 가져옴
  useEffect(() => {
    fetchWeather(); // 초기 렌더링 시 날씨 데이터를 가져옴
    const intervalId = setInterval(() => fetchWeather(), 300000); // 5분마다 갱신
    return () => clearInterval(intervalId); // 컴포넌트가 언마운트될 때 정리
  }, []);

  // 날짜를 설정하는 useEffect
  useEffect(() => {
    const date = new Date();
    const month = date.getMonth() + 1;
    const day = date.getDate();
    const weekDays = ["일", "월", "화", "수", "목", "금", "토"];
    const dayOfWeek = weekDays[date.getDay()];

    const formattedDate = `${month}월 ${day}일 (${dayOfWeek})`;
    setDate(formattedDate); // 날짜 상태 업데이트
  }, []);

  return (
    <div>
      {weather ? (
        <div className="grid-container">
          <div className="city">구미 현재 날씨</div>
          <div className="today" id="today-date">
            {date} {/* 날짜 표시 */}
          </div>
          <div className="weather">
            <img
              className="weatherIcon"
              src={weather.weather_icon}
              alt={weather.weather}
              width="100"
              height="100"
            />
            <p className="temp">맑음</p> {/* 날씨 설명 표시 */}
          </div>
          <div className="temp">
            현재<span className="temp_num">{weather.current_temp}</span>
          </div>
          <div className="temp">
            체감<span className="temp_num">{weather.feeling_temp}</span>
          </div>
          <div className="max">
            최고: <span className="max_color">{weather.max_temp}</span>
          </div>
          <div className="min">
            최저: <span className="min_color">{weather.min_temp}</span>
          </div>
        </div>
      ) : (
        <div>Loading...</div> // 로딩 상태 표시
      )}
    </div>
  );
}

export default Weather;
