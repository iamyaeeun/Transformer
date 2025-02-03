class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        super().__init__()

        self.self_atten = MHA(d_model, n_heads) # 멀티 헤드 어텐션
        self.self_atten_LN = nn.LayerNorm(d_model) # Layer Normalization

        self.FF = FeedForward(d_model, d_ff, drop_p) # FeedForward 네트워크
        self.FF_LN = nn.LayerNorm(d_model) # Layer Normalization

        self.dropout = nn.Dropout(drop_p) # 드롭아웃

    def forward(self, x, enc_mask):
        # 멀티 헤드 어텐션 적용
        residual, atten_enc = self.self_atten(x, x, x, enc_mask)
        residual = self.dropout(residual)
        x = self.self_atten_LN(x + residual)  # 잔차 더한 후 정규화

        # FeedForward 적용
        residual = self.FF(x)
        residual = self.dropout(residual)
        x = self.FF_LN(x + residual)  # 잔차 더한 후 정규화

        return x, atten_enc

class Encoder(nn.Module):
    def __init__(self, input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p):
        super().__init__()

        self.scale = torch.sqrt(torch.tensor(d_model))
        self.input_embedding = input_embedding  # 입력 임베딩
        self.pos_embedding = nn.Embedding(max_len, d_model)  # 위치 임베딩

        self.dropout = nn.Dropout(drop_p)

        # 여러 개의 인코더 레이어 생성
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, drop_p) for _ in range(n_layers)])        

    def forward(self, src, mask, atten_map_save=False):
        
        pos = torch.arange(src.shape[1]).repeat(src.shape[0], 1).to(DEVICE) # 위치 정보 생성

        # 입력 임베딩과 위치 임베딩을 더한 후 스케일링 적용
        x = self.scale * self.input_embedding(src) + self.pos_embedding(pos) # self.scale 곱해주면 position 보다 token 정보를 더 보게 됨
        x = self.dropout(x)
        
        atten_encs = torch.tensor([]).to(DEVICE)
        for layer in self.layers:
            x, atten_enc = layer(x, mask)
            if atten_map_save is True:
                atten_encs = torch.cat([atten_encs, atten_enc[0].unsqueeze(0)], dim=0)

        return x, atten_encs
