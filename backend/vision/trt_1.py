import tensorrt as trt

# TensorRT 로거 설정
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 엔진 빌더와 네트워크 생성
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1)
parser = trt.OnnxParser(network, TRT_LOGGER)

# ONNX 모델 파일 읽기
with open("ir101.onnx", "rb") as model:
    parser.parse(model.read())

# 빌더 구성
config = builder.create_builder_config()

# 워크스페이스 메모리 4GB로 설정
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))

# FP16 모드 활성화
config.set_flag(trt.BuilderFlag.FP16)

# 배치 크기 설정
builder.max_batch_size = 8

# 엔진 빌드
engine = builder.build_engine(network, config)

# 엔진 저장
with open("adaface_ir101_fp16.engine", "wb") as f:
    f.write(engine.serialize())
