# VAE   
   
VAE 기반 범죄자 변장 모습 예측 모델 연구   
    
## 1. 데이터   
   
CelebA 데이터 셋 : 대표적인 얼굴 데이터 셋   
- 약 200,000개 정도의 얼굴 이미지로 구성   
- 각 얼굴에 대해서 40개의 이진 레이블
- 10,000명가량의 사람이 포함
- 178 x 218 해상도
   
CelebA 데이터와 픽셀을 맞추고 인물을 중심을 상하좌우 여백을 비슷하게 준비   

## 2. 인코더   
   
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
   

## 3. 디코더   
   
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
        
## 4. 훈련   
   
- 얼굴의 상세 정보를 충분히 인코딩하기 위해 200개의 벡터 공간을 사용
- 재구성 손실 파라미터를 10,000으로 증가
   
* vae_train.ipynb 파일 참고   
   
## 5. 특성벡터 추출   
   
변장 모습 예측 모델 구조도   
<img src="https://user-images.githubusercontent.com/62587484/149878727-c7619142-ad01-485d-aba9-71f1d927fb37.png" width="100%">   

- 특성벡터를 추출하기 위한 특성벡터 추출경로(Latent vector extraction path)와 이미지 생성 경로(Image generation path)로 구성   
- 특성벡터 추출경로에서는 사람의 성별, 머리 색깔 등의 다양한 정보가 반영되어 있는 특성벡터를 추출하기 위한 특성벡터 추출기(Latent Vector Extractor)로 구성
- 특성벡터 추출기에서 잠재 분포를 학습하고 그 분포에서 샘플링된 값인 특성벡터가 도출   
   
#### 특성으로 향하는 벡터   
   
- (남성, 여성), (안경, 안경 미착용)과 같이 반대되는 특성을 지닌 그룹간의 인코딩 벡터의 차분을 이용하여 각각의 특성을 지니는 특성벡터를 추출   
   
<pre><code>    while(current_n_POS < 10000):

        batch = next(data_flow_label)
        im = batch[0]
        attribute = batch[1]
        
        z = vae.encoder.predict(np.array(im))

        z_POS = z[attribute==1]   
        z_NEG = z[attribute==-1]
    
        if len(z_POS) > 0:
            current_sum_POS = current_sum_POS + np.sum(z_POS, axis = 0)
            current_n_POS += len(z_POS)
            new_mean_POS = current_sum_POS / current_n_POS
            movement_POS = np.linalg.norm(new_mean_POS-current_mean_POS)

        if len(z_NEG) > 0: 
            current_sum_NEG = current_sum_NEG + np.sum(z_NEG, axis = 0)
            current_n_NEG += len(z_NEG)
            new_mean_NEG = current_sum_NEG / current_n_NEG
            movement_NEG = np.linalg.norm(new_mean_NEG-current_mean_NEG)

        current_vector = new_mean_POS-new_mean_NEG
        new_dist = np.linalg.norm(current_vector)
        dist_change = new_dist - current_dist

        current_mean_POS = np.copy(new_mean_POS)
        current_mean_NEG = np.copy(new_mean_NEG)
        current_dist = np.copy(new_dist)

        if np.sum([movement_POS, movement_NEG]) < 0.08:
            current_vector = current_vector / current_dist
            print('Found the ' + label + ' vector')
            break</code></pre>   
   
#### 입력 이미지에 특성벡터 적용   
   
- 아래 식은 인코딩된 이미지에 특성벡터를 적용하기 위한 수식으로 특성벡터의 크기를 조절할 수 있는 alpha 값이 존재   
   
<img src="https://user-images.githubusercontent.com/62587484/149879909-b837bd82-5dba-4640-abf7-e255ab02fef1.png" width="100%">   
   
- alpha는 합성되는 특성 벡터에 대한 가중치로 새로운 특성이 더해지는 정도를 조절
- 인코딩된 이미지에 특성벡터와 alpha를 곱한 값을 더하고 디코더를 거치면 특성이 더해진 새로운 이미지가 도출
- alpha가 증가할수록 원본 이미지의 얼굴이 점차 특성벡터가 더해진 얼굴이 생성   
   
<pre><code>        for factor in factors:

            changed_z_point = JC_z_points[i] + feature_vec * factor
            changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]
</code></pre>
   
## 6. 결과   
   
아래 그림은 남성 벡터를 alpha 값인 0, 1, 2, 3, 4에 따라 합성 후 디코딩한 결과   
- alpha가 0인 경우 원본 이미지에 인코딩/디코딩 과정을 수행한 결과로 원본 이미지를 재구성한 결과를 의미
   
<img src="https://user-images.githubusercontent.com/62587484/149881251-48c980d7-beda-4977-974c-520de3f94c7d.PNG" width="100%">   
   
아래 그림은 남성 벡터를 alpha 값인 –4, -3, -2, -1, 0에 따라 합성 후 디코딩한 결과   
원본 이미지에서 안경 특성벡터와 반대되는 특성벡터가 더해지면서 안경 특성이 반영되지 않는 이미지가 생성되는 결과   
   
<img src="https://user-images.githubusercontent.com/62587484/149881264-ff3e1203-3332-474f-8b75-46ed989dc90a.PNG" width="100%">   
   
   
- CelebA 데이터에서 약 40개의 특성 벡터를 추출할 수 있으며 이를 범죄자 변장 모습 예측에 사용할 수 있음
