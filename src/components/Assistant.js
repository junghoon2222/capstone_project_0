import React, { useState, useEffect, useRef } from "react";

const AudioRecorder = ({ setSiriText, setUserText }) => {
  const [isRecording, setIsRecording] = useState(true);
  const socketRef = useRef(null); // WebSocket 참조를 useRef로 변경
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const workletNodeRef = useRef(null);

  useEffect(() => {
    // WebSocket 연결 설정
    const ws = new WebSocket("ws://182.218.49.58:50007");
    socketRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket 연결 성공");
    };

    ws.onmessage = (event) => {
      const message = event.data;

      // 텍스트 메시지 처리
      if (typeof message === "string") {
        if (message.startsWith("input ")) {
          setUserText(message.slice(6));
        } else if (message.startsWith("output ")) {
          setSiriText(message.slice(7));
        }
      }
      // 바이너리 메시지 처리
      else if (message instanceof Blob) {
        // Blob 데이터를 오디오로 재생
        const audioBlob = message;
        const audioUrl = URL.createObjectURL(audioBlob);

        // 오디오 객체 생성 및 재생
        const audio = new Audio(audioUrl);
        audio.play();
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket 에러:", error);
    };

    ws.onclose = () => {
      console.log("WebSocket 연결이 종료되었습니다.");
    };

    return () => {
      // 컴포넌트가 언마운트될 때 WebSocket 연결 해제
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [setSiriText, setUserText]);

  useEffect(() => {
    // 컴포넌트 마운트 시 녹음 시작
    startRecording();

    return () => {
      // 컴포넌트 언마운트 시 녹음 중지
      stopRecording();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startRecording = async () => {
    try {
      // AudioContext 생성
      audioContextRef.current = new (window.AudioContext ||
        window.webkitAudioContext)();

      // AudioWorklet 추가 (public/audioProcessor.js 필요)
      await audioContextRef.current.audioWorklet.addModule("audioProcessor.js");

      // 마이크 접근 권한 요청 및 스트림 받기
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      // AudioContext에 MediaStreamSource 연결
      const source = audioContextRef.current.createMediaStreamSource(stream);

      // AudioWorkletNode 생성
      workletNodeRef.current = new AudioWorkletNode(
        audioContextRef.current,
        "audio-processor"
      );

      // AudioWorkletNode에서 메인 스레드로 오디오 데이터 수신
      workletNodeRef.current.port.onmessage = (event) => {
        const audioData = event.data; // Float32Array

        if (
          socketRef.current &&
          socketRef.current.readyState === WebSocket.OPEN
        ) {
          socketRef.current.send(audioData.buffer); // 버퍼로 전송
        }
      };

      // 연결: source -> workletNode
      source.connect(workletNodeRef.current);
      // 스피커로 출력하지 않기 위해 destination에 연결하지 않음

      setIsRecording(true);
    } catch (error) {
      console.error("마이크 접근 또는 오디오 워크렛 설정 중 오류 발생:", error);
    }
  };

  const stopRecording = () => {
    // 녹음 중단 및 자원 해제
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    setIsRecording(false);
  };

  return null; // UI가 필요 없으므로 null 반환
};

export default AudioRecorder;
