import React, { useState, useEffect, useRef } from "react";

import Weather from "./components/weather";
import Clock from "./components/Clock";
// import Card from "./components/Cards";
import AudioRecorder from "./components/AudioRecorder";

import "bootstrap/dist/css/bootstrap.min.css";

import "./Main.css";

// library.add(faChevronDown, faChevronUp);

function Main() {
  const [mode, setMode] = useState("active"); // 'active', 'standby', 'signup'

  const [userName, setUserName] = useState("jeonghun");
  const [userText, setUserText] = useState("대기 중..");
  const [siriText, setSiriText] = useState(
    "안녕하세요! \n 스마트 미러 어시스턴트입니다."
  );
  const audioRecorderRef = useRef();

  useEffect(() => {
    if (audioRecorderRef.current) {
      setUserText(audioRecorderRef.current.isRecording ? "녹음 중.." : "대기 중..");
    }
  }, [audioRecorderRef.current?.isRecording]);

  const renderHeader = () => (
    <div className="row">
      <div className="col text-center text-white">
        <div className="row">
          <div className="clock-weather-container">
            <div className="clockbox">
              <Clock />
            </div>
            <div className="weatherbox"></div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderCards = () => (
    
    <div className="row text-white">
      <div className="row">
        <div className="col" style={{ height: "20vh" }}></div>
      </div>
      <div className="col-5">
        <div className="col" style={{ height: "14vh" }}>
          <Weather />
        </div>
      </div>

      <div className="col-2"></div>
      <div className="col-5">
      </div>
    </div>

  );


  const renderFooter = () => (
    <div className="fixed-bottom mb-4 text-white">
      <div className="container">
        <div></div>

        <div className="fs-3 mt-4 mb-5">
          <div></div>
          <div
            className="row"
            style={{
              color: "white",

              // fontFamily: "GmarketSansLight",
              fontSize: "50px",
              // fontWeight: "bold",
              width: "100%",
              // justifyContent: "center",
              margin: "auto",
            }}
          >
            <div className="col-2"></div>
            <div className="col-8">
              <div className="notifications-container">
                <div className="success">
                  <div className="flex">
                    
                  <div className={audioRecorderRef.current?.isRecording ? "recording-circle" : "not-recording-circle"}></div>
                  <div className="success-prompt-wrap">
                      <p className="success-prompt-heading">{userText}</p>
                      <div className="success-prompt-prompt">
                        <p>{siriText}</p>
                      </div>
                      <div className="success-button-container">
                        <button type="button" className="success-button-main">
                          재전송
                        </button>
                        <button
                          type="button"
                          className="success-button-secondary"
                        >
                          초기화
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-2"></div>
          </div>
        </div>
      </div>
      <div className="container-fluid mb-1 mt-1 text-warning-1">
        <div className="row" style={{ height: "2vh" }}>
          <div className="col text-start fs-4 ms-3">{userName}</div>
          <div className="col"></div>
          <div className="col fs-3 text-start me-3 mt-0"></div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="container-fluid">
      <AudioRecorder ref={audioRecorderRef} setSiriText={setSiriText} setUserText={setUserText} />

      {renderHeader()}
      {renderCards()}
      {renderFooter()}

    </div>
  );
}

export default Main;