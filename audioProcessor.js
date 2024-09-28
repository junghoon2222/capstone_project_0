class AudioProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input) {
      const channelData = input[0]; // Float32Array

      // 여기서 채널 데이터를 웹소켓으로 전송하는 로직을 추가할 수 없습니다.
      // AudioProcessor는 독립된 오디오 프로세싱 스레드에서 실행되기 때문에,
      // 데이터를 AudioWorkletNode로 보내서 처리해야 합니다.

      this.port.postMessage(channelData); // 메인 스레드로 데이터 전송
    }
    return true;
  }
}

registerProcessor("audio-processor", AudioProcessor);
