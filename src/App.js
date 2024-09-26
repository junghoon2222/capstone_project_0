import React, { useState, useEffect, useRef } from "react";
import { library } from "@fortawesome/fontawesome-svg-core";
import { faChevronDown, faChevronUp } from "@fortawesome/free-solid-svg-icons";
import { PorcupineWorkerFactory } from "@picovoice/porcupine-web";

import Weather from "./components/weather";
import Clock from "./components/Clock";
import Card from "./components/Cards";
import Assistant from "./components/Assistant";

import "bootstrap/dist/css/bootstrap.min.css";

import "./App.css";

library.add(faChevronDown, faChevronUp);

function App() {
  const [mode, setMode] = useState("active"); // 'active', 'standby', 'signup'

  const [userName, setUserName] = useState("wjdgns's mirror");
  const [data, setData] = useState({});
  const [userText, setUserText] = useState("녹음 중..");
  const [siriText, setSiriText] = useState(
    "안녕하세요! \n 스마트 미러 어시스턴트입니다."
  );

  const websocketRef = useRef(null);

  useEffect(() => {
    let reconnectTimeout;
    const MAX_RETRIES = 100;
    const RECONNECT_INTERVAL = 3000;
    let retryCount = 0;

    const connectWebSocket = () => {
      websocketRef.current = new WebSocket("ws://182.218.49.58:50007");

      websocketRef.current.onopen = () => {
        console.log("Assistant WebSocket Connected");
        retryCount = 0;
      };

      websocketRef.current.onmessage = (event) => {
        const message = event.data;

        if (message.startsWith("input ")) {
          setUserText(message.slice(6));
        } else if (message.startsWith("output ")) {
          setSiriText(message.slice(7));
        }
      };

      websocketRef.current.onclose = () => {
        console.log("Assistant WebSocket Disconnected");
        attemptReconnect();
      };

      websocketRef.current.onerror = (error) => {
        console.log("Assistant WebSocket Error Occurred: ", error);
        websocketRef.current.close();
      };
    };

    const attemptReconnect = () => {
      if (retryCount < MAX_RETRIES) {
        console.log(
          `Trying to Reconnect... (${retryCount + 1}/${MAX_RETRIES})`
        );
        retryCount += 1;
        reconnectTimeout = setTimeout(connectWebSocket, RECONNECT_INTERVAL);
      } else {
        console.log("Max Reconnecting Tried");
      }
    };

    connectWebSocket();

    return () => {
      if (websocketRef.current) websocketRef.current.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, []);

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
        <div className="col" style={{ height: "30vh" }}></div>
      </div>
      <div className="col-6">
        <div className="col" style={{ height: "18vh" }}>
          <Weather />
        </div>
      </div>

      <div className="col-1"></div>
      <div className="col-5">
        <Card />
      </div>
    </div>
  );

  const renderSignup = () => (
    <div className="row">
      <div className="col text-center py-3 text-white">
        <h1>Signup Form</h1>
        {/* 나중에 signup 폼을 구현할 예정 */}
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
              <div class="notifications-container">
                <div class="success">
                  <div class="flex">
                    <div class="recording-circle"></div>
                    <div class="success-prompt-wrap">
                      <p class="success-prompt-heading">{userText}</p>
                      <div class="success-prompt-prompt">
                        <p>{siriText}</p>
                      </div>
                      <div class="success-button-container">
                        <button type="button" class="success-button-main">
                          재전송
                        </button>
                        <button type="button" class="success-button-secondary">
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
      <Assistant websocket={websocketRef.current} />

      {(mode === "active" || mode === "signup") && renderHeader()}
      {mode === "active" && renderCards()}
      {mode === "signup" && renderSignup()}
      {mode === "active" && renderFooter()}
      {mode === "standby" && (
        <div className="d-flex flex-column justify-content-center align-items-center vh-100">
          <Clock />
          <Weather />
        </div>
      )}
    </div>
  );
}

export default App;
