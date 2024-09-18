// App.js

import React, { useEffect } from "react";

function App() {
  useEffect(() => {
    const threshold = 0.02; // 음성 감지 임계값 (조정 필요)
    const silenceDuration = 1000; // 침묵 시간 (밀리초)
    let audioContext;
    let analyser;
    let microphone;
    let javascriptNode;
    let audioData = [];
    let isRecording = false;
    let silenceStart = null;

    const websocket = new WebSocket("ws://182.218.49.58:50007");

    websocket.onmessage = (event) => {
      const response = event.data;
      if (!filterText(response)) {
        console.log("Server Response:", response);
      }
    };

    // 마이크 접근 및 오디오 처리 설정
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        microphone = audioContext.createMediaStreamSource(stream);
        javascriptNode = audioContext.createScriptProcessor(4096, 1, 1);

        analyser.smoothingTimeConstant = 0.8;
        analyser.fftSize = 1024;

        microphone.connect(analyser);
        analyser.connect(javascriptNode);
        javascriptNode.connect(audioContext.destination);

        javascriptNode.onaudioprocess = (event) => {
          const array = new Float32Array(analyser.fftSize);
          analyser.getFloatTimeDomainData(array);

          const frameMean =
            array.reduce((sum, value) => sum + Math.abs(value), 0) /
            array.length;
          console.log("Frame Mean:", frameMean);

          if (frameMean > threshold) {
            if (!isRecording) {
              isRecording = true;
              audioData = [];
              console.log("Recording started");
            }
            audioData.push(...array);
            silenceStart = null;
          } else if (isRecording) {
            if (!silenceStart) {
              silenceStart = Date.now();
            } else if (Date.now() - silenceStart > silenceDuration) {
              isRecording = false;
              console.log("Recording stopped");
              sendAudioData(audioData);
              audioData = [];
            } else {
              audioData.push(...array);
            }
          }
        };
      })
      .catch((error) => {
        console.error("마이크 접근 에러:", error);
      });

    // 컴포넌트 언마운트 시 리소스 정리
    return () => {
      if (microphone) microphone.disconnect();
      if (analyser) analyser.disconnect();
      if (javascriptNode) javascriptNode.disconnect();
      if (audioContext) audioContext.close();
      if (websocket) websocket.close();
    };
  }, []);

  // 오디오 데이터를 서버로 전송
  const sendAudioData = (audioData) => {
    // Float32Array를 Int16Array로 변환
    const int16Array = floatTo16BitPCM(audioData);
    // ArrayBuffer를 Blob으로 변환
    const blob = new Blob([int16Array.buffer], {
      type: "application/octet-stream",
    });
    websocket.send(blob);
  };

  // Float32Array를 Int16Array로 변환하는 함수
  const floatTo16BitPCM = (input) => {
    const buffer = new ArrayBuffer(input.length * 2);
    const output = new DataView(buffer);
    for (let i = 0; i < input.length; i++) {
      let s = Math.max(-1, Math.min(1, input[i]));
      output.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Int16Array(buffer);
  };

  // 텍스트 필터링 함수
  const filterText = (text) => {
    if (!text.trim()) {
      return true;
    }
    if (containsJapanese(text) && containsEnglish(text)) {
      return true;
    }
    if (containsJapanese(text)) {
      return true;
    }
    if (containsEnglish(text)) {
      return true;
    } else {
      return false;
    }
  };

  // 일본어 포함 여부 확인 함수
  const containsJapanese = (text) => {
    const japanesePattern = /[\u3040-\u30FF\u4E00-\u9FFF]/;
    return japanesePattern.test(text);
  };

  // 영어 포함 여부 확인 함수
  const containsEnglish = (text) => {
    const englishPattern = /[A-Za-z]/;
    return englishPattern.test(text);
  };

  return (
    <div>
      <h1>Whisp Transcription</h1>
      <p>마이크 입력을 감지하고 있습니다...</p>
    </div>
  );
}

export default App;
