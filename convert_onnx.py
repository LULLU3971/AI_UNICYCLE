import torch
from stable_baselines3 import PPO
import onnx
from environment import UnicycleEnv

# 환경 생성
env = UnicycleEnv()

try:
    # 모델 로드
    model = PPO.load("final_model.zip")
    print("모델 로드 성공")
    
    # 정책 네트워크를 평가 모드로 설정
    policy = model.policy.eval()
    
    # MLP 구조 확인
    print("\nMLP 구조:")
    print(f"입력 크기: {policy.mlp_extractor.shared_net[0].in_features}")
    print(f"출력 크기: {policy.action_net.out_features}")
    
    # ONNX 내보내기를 위한 래퍼 클래스 수정
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            
        def forward(self, x):
            # 공유 네트워크를 통과
            shared_features = self.policy.mlp_extractor.shared_net(x)
            # 액터 네트워크를 통과
            actor_features = self.policy.mlp_extractor.policy_net(shared_features)
            # 최종 액션 로짓 계산
            action_logits = self.policy.action_net(actor_features)
            return action_logits
    
    # 더미 입력 생성
    obs_shape = env.observation_space.shape
    dummy_input = torch.zeros((1,) + obs_shape, dtype=torch.float32)
    
    # 테스트 실행
    wrapped_policy = PolicyWrapper(policy)
    with torch.no_grad():
        test_output = wrapped_policy(dummy_input)
        print(f"\n테스트 출력:")
        print(f"Shape: {test_output.shape}")
        print(f"Values: {test_output.detach().numpy()}")
    
    # ONNX 내보내기
    torch.onnx.export(
        wrapped_policy,
        dummy_input,
        "model.onnx",
        input_names=['observation'],
        output_names=['action_logits'],  # 출력 이름 변경
        opset_version=12,
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action_logits': {0: 'batch_size'}
        },
        verbose=True
    )
    print("\nONNX 모델 내보내기 성공")
    
    # 모델 검증
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
    print("\n모델 구조:")
    for input in onnx_model.graph.input:
        print(f"입력: {input.name}, shape: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
    for output in onnx_model.graph.output:
        print(f"출력: {output.name}, shape: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")

except Exception as e:
    print(f"\n오류 발생: {e}")
    import traceback
    print(traceback.format_exc())
    raise