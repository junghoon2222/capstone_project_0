import React, { useState, useEffect, useRef, forwardRef, useImperativeHandle } from "react";

const AudioRecorder = forwardRef(({ setSiriText, setUserText }, ref) => {
  const [isRecording, setIsRecording] = useState(false);
  const [socket, setSocket] = useState(null);
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const workletNodeRef = useRef(null);
  const audioSourceRef = useRef(null); // 기존 source를 참조하기 위한 ref

  useEffect(() => {
    // 컴포넌트가 마운트될 때 WebSocket 연결 설정
    const ws = new WebSocket("ws://182.218.49.58:50007");
    setSocket(ws);

    ws.onmessage = async (event) => {
      const message = event.data;
      
      if (typeof message === "string") {
        if (message.startsWith("input ")) { 
          setUserText(message.slice(6));
        } else if (message.startsWith("output ")) {
          setSiriText(message.slice(7));
        }
      } else if (message instanceof Blob) {
        // 바이너리 데이터를 Blob으로 받았을 때 처리
        const arrayBuffer = await message.arrayBuffer();
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // 기존의 source 중지
        if (audioSourceRef.current) {
          audioSourceRef.current.stop();
        }

        // 새로운 source 생성 및 재생
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start();

        // 새로운 source를 ref에 저장
        audioSourceRef.current = source;
      }
    };

    return () => {
      // 컴포넌트가 언마운트될 때 WebSocket 연결 해제
      if (socket) {
        socket.close();
      }
    };
  }, []);

  useImperativeHandle(ref, () => ({
    isRecording, // 부모 컴포넌트에서 접근할 수 있도록 isRecording 상태를 노출
  }));

  const startRecording = async () => {
    try {
      // AudioContext 생성
      audioContextRef.current = new (window.AudioContext ||
        window.webkitAudioContext)();

      // AudioWorklet 추가 (public/audioProcessor.js)
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

        if (socket && socket.readyState === WebSocket.OPEN) {
          socket.send(audioData.buffer); // 버퍼로 전송
        }
      };

      // 연결: source -> workletNode -> destination
      source.connect(workletNodeRef.current);
      workletNodeRef.current.connect(audioContextRef.current.destination);

      setIsRecording(true);
    } catch (error) {
      console.error(
        "Error accessing microphone or setting up audio worklet:",
        error
      );
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

  return (
    <div>
      <button onClick={isRecording ? stopRecording : startRecording}>
        {isRecording ? "Stop Recording" : "Start Recording"}
      </button>
    </div>
  );
});

export default AudioRecorder;