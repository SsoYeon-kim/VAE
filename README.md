# VAE   
    
## 1. 인코더   
   
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
   
<pre><code>        ### 인코더
        # 인코더 입력(이미지) 정의
        encoder_input = Input(shape=self.input_dim, name='encoder_input')

        x = encoder_input
        
        # 합성곱층 쌓기
        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(
                filters = self.encoder_conv_filters[i]
                , kernel_size = self.encoder_conv_kernel_size[i]
                , strides = self.encoder_conv_strides[i]
                , padding = 'same'
                , name = 'encoder_conv_' + str(i)
                )

            x = conv_layer(x)
            
            # 훈련 속도를 높이기 위해 각 합성곱 층 뒤에 배치 정규화 층
            if self.use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)
            # Dropout층 
            if self.use_dropout:
                x = Dropout(rate = 0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]
        
        # 마지막 합성곱층을 하나의 벡터로 펼침
        x = Flatten()(x)
        # flatten층을 mu츨과 log_var 층에 연결
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)
        
        # 이미지가 입력되면 mu와 log_var의 값을 출력
        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon
        
        # 람다 층이 잠재 공간에서 mu화 log_var로 정의되는 정규분포로부터 포인트 z를 샘플링 함
        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)</code></pre>
   

## 2. 디코더   
   
합성곱 층을 제외하면 인코더의 반대   
전치 합성곱(convolutional transpose)층 사용   
   
- 전치 합성곱은 표준 합성곱 층의 원리와 동일하게 필터가 이미지 위를 지나가지만 strides=2일 때 텐서의 높이와 너비를 두 배로 늘림
- 케라스에서는 Conv2DTranspose 층을 사용해 전치 합성곱 연상르 수행
- strides=2로 Conv2DTranspose 층을 쌓으면 점진적으로 크기가 증가되어 원본 이미지 차원인 28 × 28까지 되돌릴 수 있음
   
<pre><code>        ### 디코터
        
        # 입력(잠재 공간의 포인트) 정의
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        
        # 입력을 Dense 층에 연결
        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        # 전치 합성곱 층에 주입할 수 있도록 벡터를 다차원 텐서로 바꿈
        x = Reshape(shape_before_flattening)(x)
        
        # 전치 합성곱 쌓기
        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters = self.decoder_conv_t_filters[i]
                , kernel_size = self.decoder_conv_t_kernel_size[i]
                , strides = self.decoder_conv_t_strides[i]
                , padding = 'same'
                , name = 'decoder_conv_t_' + str(i)
                )

            x = conv_t_layer(x)

            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate = 0.25)(x)
            else:
                x = Activation('sigmoid')(x)

            

        decoder_output = x
        
        # 디코더를 정의하는 케라스 모델 생성
        # (잠재 공간의 한 포인트를 받아 원본 이미지 차원으로 디코딩 함)
        self.decoder = Model(decoder_input, decoder_output)

        ### VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)</code></pre>   
        
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
        
### 3. 훈련   
   

        
