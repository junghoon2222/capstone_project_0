// Assistant.js

import { useEffect, useRef, useCallback } from "react";

function Assistant({ websocket }) {
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const silenceStartRef = useRef(null);
  const isRecordingRef = useRef(false);
  const audioContextRef = useRef(null);

  const sendAudioData = useCallback(
    (audioBlob) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const arrayBuffer = reader.result;
        if (websocket && websocket.readyState === WebSocket.OPEN) {
          websocket.send(arrayBuffer);
        } else {
          console.error("WebSocket is not open");
        }
      };
      reader.readAsArrayBuffer(audioBlob);
    },
    [websocket]
  );

  useEffect(() => {
    const threshold = 0.02;
    const silenceDuration = 1000;

    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        mediaRecorderRef.current = new MediaRecorder(stream, {
          mimeType: "audio/webm;codecs=opus",
        });

        mediaRecorderRef.current.onerror = (event) => {
          console.error("MediaRecorder error:", event.error);
        };

        audioContextRef.current = new (window.AudioContext ||
          window.webkitAudioContext)();

        const source = audioContextRef.current.createMediaStreamSource(stream);
        const analyser = audioContextRef.current.createAnalyser();
        source.connect(analyser);

        const dataArray = new Float32Array(analyser.fftSize);

        // 오디오 처리 함수
        const processAudio = () => {
          analyser.getFloatTimeDomainData(dataArray);
          const frameMean =
            dataArray.reduce((sum, value) => sum + Math.abs(value), 0) /
            dataArray.length;
          console.log("Frame Mean:", frameMean);

          if (frameMean > threshold) {
            if (!isRecordingRef.current) {
              isRecordingRef.current = true;
              audioChunksRef.current = [];
              mediaRecorderRef.current.start();
              console.log("Recording started");
            }
            silenceStartRef.current = null;
          } else if (isRecordingRef.current) {
            if (!silenceStartRef.current) {
              silenceStartRef.current = Date.now();
            } else if (Date.now() - silenceStartRef.current > silenceDuration) {
              isRecordingRef.current = false;
              mediaRecorderRef.current.stop();
              console.log("Recording stopped");
            }
          }
          requestAnimationFrame(processAudio);
        };

        mediaRecorderRef.current.ondataavailable = (event) => {
          audioChunksRef.current.push(event.data);
        };

        mediaRecorderRef.current.onstop = () => {
          const audioBlob = new Blob(audioChunksRef.current, {
            type: mediaRecorderRef.current.mimeType,
          });
          sendAudioData(audioBlob);
        };

        processAudio();
      })
      .catch((error) => {
        console.error("마이크 접근 에러:", error);
      });

    return () => {
      if (audioContextRef.current) audioContextRef.current.close();
    };
  }, [sendAudioData]);

  return null;
}

export default Assistant;
