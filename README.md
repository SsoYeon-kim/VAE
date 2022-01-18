# VAE   
    
## 인코더   
   
AE에서는 각 이미지가 잠재 공간의 한 포인트에 직접 매핑   
VAE는 각 이미지가 잠재 공간에 있는 **포인트 주변의 다변수 정규 분포**에 매핑   
   
인코더는 입력 이미지를 받아 잠재 공간의 다변수 정규 분포를 정의하는 2개의 벡터
- mu, log_var로 인코딩
- mu : 이 분포의 평균 벡터
- log_var : 차원별 분산의 로그값
   
이미지를 잠재 공간의 특정 포인트 z로 인코딩하기 위해 아래식을 이용해 이 분포에서 샘플링함   
- z = mu + sigma * epsilon
- sigma = exp(log_var / 2)
- (epsilon : 표준 정규 분포에서 샘플링된 값)
   
### 손실 함수   
   
RMSE에 KL발산(쿨백-라이블러 발산)추가   
- KL발산 : 한 확률 분포가 다른 분포와 얼마나 다른지 측정
- 포인트를 선택할 때 사용할 수 있는 잘 정의된 분포(표준 정규 분포)를 가지게 됨
- 모든 인코딩된 분포를 표준 정규 분포에 가깝게 되도록 강제함 (포인트 군집 사이에 큰 간격이 생길 가능성이 적음)
   
<pre><code>    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        # RMSE(재구성 손실)
        # r_loss_factor로 가중치를 주어 KL발산 손실과 균형을 맞춤
        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
            return r_loss_factor * r_loss
        # KL발산 추가 (쿨백- 라이블러 발산)
        def vae_kl_loss(y_true, y_pred):
            kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return  r_loss + kl_loss

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])</code></pre>
