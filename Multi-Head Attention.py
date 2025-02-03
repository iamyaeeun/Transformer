class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model  # 모델의 차원
        self.n_heads = n_heads  # 멀티 헤드 개수
        self.head_dim = int(d_model / n_heads)  # 각 헤드의 차원

        # Query, Key, Value 구하기 위한 선형 변환 레이어
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)  # 최종 출력 변환 레이어

        self.scale = torch.sqrt(torch.tensor(self.head_dim))  # 스케일링 위한 값 (루트 d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        # 입력 텐서에 선형 변환 적용해 Query, Key, Value 구함
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        # 멀티 헤드로 변환 (배치 크기, 시퀀스 길이, 헤드 개수, 헤드 차원) -> (배치 크기, 헤드 개수, 시퀀스 길이, 헤드 차원)
        Q = Q.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # 어텐션 점수 계산 (Q x K^T) / sqrt(d_k)
        attention_score = Q @ K.permute(0, 1, 3, 2) / self.scale

        # 마스크 적용
        if mask is not None:
            attention_score[mask] = -1e10  # 마스크된 위치에 매우 작은 값 적용 (softmax에서 0에 가까운 값으로 만듦)
        
        # 소프트맥스 사용하여 어텐션 분포 생성
        attention_dist = torch.softmax(attention_score, dim=-1)

        # 어텐션 가중치 적용한 값 계산
        attention = attention_dist @ V

        # 원래 차원으로 변환 (배치 크기, 시퀀스 길이, 헤드 개수, 헤드 차원) -> (배치 크기, 시퀀스 길이, d_model)
        x = attention.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, -1, self.d_model)
        
        # 최종 출력 선형 변환 적용
        x = self.fc_o(x)
        return x, attention_dist

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()

        # MLP
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),  # 활성화 함수
            nn.Dropout(drop_p),  # 드롭아웃
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        return self.linear(x)
