class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop_p):
        super().__init__()

        self.self_atten = MHA(d_model, n_heads) # 셀프 어텐션
        self.self_atten_LN = nn.LayerNorm(d_model) # Layer Normalization
        
        self.enc_dec_atten = MHA(d_model, n_heads) # 인코더 디코더 어텐션
        self.enc_dec_atten_LN = nn.LayerNorm(d_model)

        self.FF = FeedForward(d_model, d_ff, drop_p) # FeedForward 네트워크
        self.FF_LN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, enc_out, dec_mask, enc_dec_mask):

        residual, atten_dec = self.self_atten(x, x, x, dec_mask) # 셀프 어텐션 적용
        residual = self.dropout(residual)
        x = self.self_atten_LN(x + residual)

        residual, atten_enc_dec = self.enc_dec_atten(x, enc_out, enc_out, enc_dec_mask) # 인코더 디코더 어텐션 적용(Q는 디코더로부터 K,V는 인코더로부터)
        residual = self.dropout(residual)
        x = self.enc_dec_atten_LN(x + residual)

        residual = self.FF(x)
        residual = self.dropout(residual)
        x = self.FF_LN(x + residual)

        return x, atten_dec, atten_enc_dec

class Decoder(nn.Module):
    def __init__(self, input_embedding, max_len, d_model, n_heads, n_layers, d_ff, drop_p):
        super().__init__()

        self.scale = torch.sqrt(torch.tensor(d_model))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, drop_p) for _ in range(n_layers)])

        self.fc_out = nn.Linear(d_model, vocab_size)
        # self.fc_out = fc_out

    def forward(self, trg, enc_out, dec_mask, enc_dec_mask, atten_map_save = False):

        pos = torch.arange(trg.shape[1]).repeat(trg.shape[0], 1).to(DEVICE)

        x = self.scale*self.input_embedding(trg) + self.pos_embedding(pos) # self.scale 곱해주면 position 보다 token 정보를 더 보게 됨
        x = self.dropout(x)

        atten_decs = torch.tensor([]).to(DEVICE)
        atten_enc_decs = torch.tensor([]).to(DEVICE)
        for layer in self.layers:
            x, atten_dec, atten_enc_dec = layer(x, enc_out, dec_mask, enc_dec_mask)
            if atten_map_save is True:
                atten_decs = torch.cat([atten_decs , atten_dec[0].unsqueeze(0)], dim=0)
                atten_enc_decs = torch.cat([atten_enc_decs , atten_enc_dec[0].unsqueeze(0)], dim=0)

        x = self.fc_out(x)
        
        return x, atten_decs, atten_enc_decs
