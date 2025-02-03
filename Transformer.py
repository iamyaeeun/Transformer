class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_heads, n_layers, d_ff, drop_p):
        super().__init__()

        input_embedding = nn.Embedding(vocab_size, d_model) 
        self.encoder = Encoder(input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p) # 인코더 초기화
        self.decoder = Decoder(input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p) # 디코더 초기화

        self.n_heads = n_heads

        for m in self.modules():
            if hasattr(m,'weight') and m.weight.dim() > 1: # 인풋 임베딩은 그대로 쓰기 위함 
                nn.init.xavier_uniform_(m.weight) # 가중치 초기화 (Xavier 초기화 사용)

    def make_enc_mask(self, src): # 인코더 마스크 생성 (패딩 토큰 마스킹)
        
        enc_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
        enc_mask = enc_mask.repeat(1, self.n_heads, src.shape[1], 1) 

        return enc_mask

    def make_dec_mask(self, trg): # 디코더 마스크 생성 (패딩 마스킹 + 미래 단어 마스킹)

        trg_pad_mask = (trg.to('cpu') == pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = trg_pad_mask.repeat(1, self.n_heads, trg.shape[1], 1)

        trg_dec_mask = torch.tril(torch.ones(trg.shape[0], self.n_heads, trg.shape[1], trg.shape[1]))==0

        dec_mask = trg_pad_mask | trg_dec_mask

        return dec_mask

    def make_enc_dec_mask(self, src, trg): # 인코더-디코더 마스크 생성 (패딩 마스킹)

        enc_dec_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
        enc_dec_mask = enc_dec_mask.repeat(1, self.n_heads, trg.shape[1], 1)

        return enc_dec_mask

    def forward(self, src, trg): # 순전파 연산 수행

        enc_mask = self.make_enc_mask(src) # 인코더 마스크 생성
        dec_mask = self.make_dec_mask(trg) # 디코더 마스크 생성
        enc_dec_mask = self.make_enc_dec_mask(src, trg) # 인코더-디코더 마스크 생성

        enc_out, atten_encs = self.encoder(src, enc_mask) # 인코더 실행
        out, atten_decs, atten_enc_decs = self.decoder(trg, enc_out, dec_mask, enc_dec_mask) # 디코더 실행

        return out, atten_encs, atten_decs, atten_enc_decs # 모델 출력과 attention map 반환
